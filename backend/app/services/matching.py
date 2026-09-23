"""マッチングのオーケストレーション: パッチ → 特徴 → OT → 計画の要約 (SPEC 2 / 3)。

FastAPI に依存しない (CLI やテストから直接呼べる)。API 向けのエラーは ``AppError`` で送出する。
各段階の所要時間は ``stats.elapsed_ms`` に入る。GPU の処理は非同期なので、計測の前後で同期する。
"""

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from typing import Literal

import torch
from PIL import Image
from torch import Tensor

from app.errors import AppError
from app.ot import BackendUnavailableError, SinkhornBackend, SolveInfo, SolveParams, get_backend
from app.ot.hard import HardAssignment, summarize_hard
from app.ot.plan import PlanSummary, summarize_plan
from app.schemas.common import ErrorCode
from app.services.deep import ModelLoadError, deep_available
from app.services.device import get_device_info
from app.services.features import FeatureParams, build_features
from app.services.notices import Notice, WarningCode
from app.services.patches import (
    GridMeta,
    compute_grid,
    extract_patches,
    image_to_tensor,
    patch_centers,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MatchParams:
    patch_size: int = 16
    stride: int | None = None  # None なら patch_size
    feature: FeatureParams = field(default_factory=FeatureParams)
    solve: SolveParams = field(default_factory=SolveParams)
    backend: Literal["auto", "flash", "dense"] = "auto"
    top_k: int = 3
    min_weight: float = 1e-4
    compute_divergence: bool = True

    @property
    def effective_stride(self) -> int:
        return self.patch_size if self.stride is None else self.stride


@dataclass(frozen=True)
class MatchStats:
    n_patches_a: int
    n_patches_b: int
    feature_dim: int
    eps: float
    sinkhorn_divergence: float | None
    backend: str
    device: str
    iterations: int
    converged: bool
    row_mass_error: float | None
    hard_agreement: float  # ハードの対応先がソフトの top-1 と一致した割合
    elapsed_ms: dict[str, float]


@dataclass(frozen=True)
class MatchResult:
    grid_a: GridMeta
    grid_b: GridMeta
    plan: PlanSummary  # CPU 上
    hard: HardAssignment  # CPU 上
    stats: MatchStats
    warnings: list[Notice]


def match_images(
    img_a: Image.Image,
    img_b: Image.Image,
    params: MatchParams,
    *,
    max_patches: int,
    warn_row_mass_error: float = 0.05,
) -> MatchResult:
    """前処理済み RGB 画像 2 枚から、A の各パッチの B への対応を求める。

    ``max_patches`` は A・B それぞれのパッチ数の上限。``stats.elapsed_ms.preprocess`` は
    パッチ抽出の時間 (画像のデコードは含まない)。
    """
    backend = _get_backend(params.backend)
    stride = params.effective_stride
    grid_a = compute_grid(img_a.width, img_a.height, params.patch_size, stride)
    grid_b = compute_grid(img_b.width, img_b.height, params.patch_size, stride)
    _check_patch_limit(img_a.size, img_b.size, grid_a, grid_b, params, max_patches)
    deep = params.feature.type == "dinov2"
    if deep and not deep_available():
        raise AppError(
            ErrorCode.INVALID_PARAMS,
            "Feature type 'dinov2' requires the optional 'deep' dependencies (timm).",
            status_code=422,
            hint="Install them with `uv sync --extra deep` and restart the server.",
        )

    device = torch.device(get_device_info().device)
    timings: dict[str, float] = {}
    warnings: list[Notice] = []
    started = time.perf_counter()

    with _stage(timings, "preprocess", device):
        tensor_a, tensor_b = image_to_tensor(img_a), image_to_tensor(img_b)
        patches_a, _ = extract_patches(tensor_a, params.patch_size, stride)
        patches_b, _ = extract_patches(tensor_b, params.patch_size, stride)

    n, m = grid_a.n, grid_b.n
    sp = params.solve

    # 特徴 (DINOv2・PCA も GPU で動く) からハード割当までを、OOM・CUDA エラーの変換の対象にする
    try:
        with _stage(timings, "features", device):
            images = (tensor_a, tensor_b) if deep else None
            feats = build_features(
                patches_a, grid_a, patches_b, grid_b, params.feature, device, images=images
            )
            warnings += feats.warnings

        a = torch.full((n,), 1.0 / n, device=device)
        b = torch.full((m,), 1.0 / m, device=device)
        with _stage(timings, "solve", device):
            f, g, eps, info = backend.solve_potentials(feats.x, feats.y, a, b, sp)
        _require_finite("potentials", f, g)

        divergence = None
        timings["divergence"] = 0.0  # 計算しないときは同期の待ち時間も載せない
        if params.compute_divergence:
            with _stage(timings, "divergence", device):
                divergence = backend.divergence(feats.x, feats.y, a, b, sp)

        centers_a, centers_b = patch_centers(grid_a, device), patch_centers(grid_b, device)
        with _stage(timings, "plan", device):
            plan = summarize_plan(
                feats.x, feats.y, f, g, a, b, eps, centers_a, centers_b,
                cost_scale=sp.cost_scale, top_k=params.top_k, min_weight=params.min_weight,
            )  # fmt: skip
            plan = _to_cpu(plan)

        with _stage(timings, "hard", device):
            hard_idx = backend.c_transform(feats.x, feats.y, g, sp)
            hard = summarize_hard(
                hard_idx, feats.x, feats.y, a, b, centers_a, centers_b, cost_scale=sp.cost_scale
            )
            hard = _to_cpu(hard)
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        raise AppError(
            ErrorCode.SOLVER_FAILED,
            "Out of GPU memory while solving.",
            status_code=500,
            hint="Use fewer patches (larger patch size) or the dense backend on smaller inputs.",
        ) from e
    except ModelLoadError as e:
        logger.exception("failed to load the DINOv2 model")
        raise AppError(
            ErrorCode.INTERNAL_ERROR,
            f"Failed to load the DINOv2 model: {e}",
            status_code=503,
            hint="The first run downloads the weights from Hugging Face Hub; check network access.",
        ) from e
    except BackendUnavailableError as e:
        raise _backend_unavailable(e) from e
    except RuntimeError as e:  # Triton のコンパイル失敗、CUDA エラーなど
        logger.exception("solver failed")
        raise AppError(
            ErrorCode.SOLVER_FAILED,
            f"The solver failed: {e}",
            status_code=500,
            hint="Try backend='dense' or different parameters.",
        ) from e

    _require_finite(
        "transport plan summary",
        plan.row_mass, plan.col_mass, plan.entropy, plan.displacement, plan.top_weight,
    )  # fmt: skip

    row_mass_error = _row_mass_error(plan, n) if sp.balanced else None
    warnings += _convergence_warnings(info, sp, row_mass_error, warn_row_mass_error)

    timings["total"] = (time.perf_counter() - started) * 1000
    stats = MatchStats(
        n_patches_a=n,
        n_patches_b=m,
        feature_dim=feats.dim,
        eps=eps,
        sinkhorn_divergence=divergence,
        backend=info.backend,
        device=info.device,
        iterations=info.iterations,
        converged=info.converged,
        row_mass_error=row_mass_error,
        hard_agreement=float((hard.idx == plan.top_idx[:, 0]).float().mean()),
        elapsed_ms={k: round(v, 2) for k, v in timings.items()},
    )
    return MatchResult(grid_a, grid_b, plan, hard, stats, warnings)


def _get_backend(setting: Literal["auto", "flash", "dense"]) -> SinkhornBackend:
    try:
        return get_backend(setting)
    except BackendUnavailableError as e:
        raise _backend_unavailable(e) from e


def _backend_unavailable(e: BackendUnavailableError) -> AppError:
    return AppError(
        ErrorCode.INVALID_PARAMS,
        f"The requested backend is not available: {e}",
        status_code=422,
        hint="Use backend 'auto' or 'dense'.",
    )


def _check_patch_limit(
    size_a: tuple[int, int],
    size_b: tuple[int, int],
    grid_a: GridMeta,
    grid_b: GridMeta,
    params: MatchParams,
    max_patches: int,
) -> None:
    if max(grid_a.n, grid_b.n) <= max_patches:
        return
    # patch_size = stride を大きくしていき、両画像が収まる最小の値を推奨する
    suggestion = None
    # 現在の size から探す (stride < size で超過したなら、同じ size を stride = size にすれば収まりうる)
    for size in range(params.patch_size, min(*size_a, *size_b) + 1):
        if all(compute_grid(w, h, size, size).n <= max_patches for w, h in (size_a, size_b)):
            suggestion = size
            break
    hint = (
        f"Use patch size {suggestion} (stride {suggestion}) or larger."
        if suggestion
        else "Use smaller images."
    )
    raise AppError(
        ErrorCode.TOO_MANY_PATCHES,
        f"Too many patches: A={grid_a.n}, B={grid_b.n} (limit {max_patches} each).",
        status_code=422,
        hint=hint,
    )


def _require_finite(what: str, *tensors: Tensor) -> None:
    if all(bool(torch.isfinite(t).all()) for t in tensors):
        return
    raise AppError(
        ErrorCode.SOLVER_FAILED,
        f"The solver diverged: non-finite values in the {what}.",
        status_code=500,
        hint="Increase blur, or use feature normalization 'zscore' or 'l2'.",
    )


def _to_cpu[T: (PlanSummary, HardAssignment)](obj: T) -> T:
    return type(obj)(**{f.name: getattr(obj, f.name).cpu() for f in fields(obj)})


def _row_mass_error(plan: PlanSummary, n: int) -> float:
    """``max_i |row_mass_i / a_i - 1|`` (``a_i = 1/N``)。"""
    return float((plan.row_mass * n - 1).abs().max())


def _convergence_warnings(
    info: SolveInfo, sp: SolveParams, row_mass_error: float | None, tolerance: float
) -> list[Notice]:
    warnings = []
    if row_mass_error is not None and row_mass_error > tolerance:
        state = "converged" if info.converged else "stopped at its iteration limit"
        warnings.append(
            Notice(
                WarningCode.ROW_MASS_ERROR,
                f"Row masses deviate from the uniform weights by up to {row_mass_error:.0%} "
                f"(the solver {state}). Weights are normalised per row so the display remains "
                "valid, but the marginals are inexact; try a larger blur.",
                {"error": row_mass_error, "tolerance": tolerance, "converged": info.converged},
            )
        )
    elif row_mass_error is None and not info.converged:
        warnings.append(
            Notice(
                WarningCode.NOT_CONVERGED,
                f"The solver stopped at its iteration limit ({info.iterations} iterations) "
                "before reaching the threshold; the result may be approximate.",
                {"iterations": info.iterations},
            )
        )
    return warnings


@contextmanager
def _stage(timings: dict[str, float], name: str, device: torch.device) -> Iterator[None]:
    _sync(device)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _sync(device)
        timings[name] = (time.perf_counter() - t0) * 1000


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
