"""パッチ → 特徴ベクトル (SPEC 3.3)。

処理順: 基本特徴 (raw / pca / color / dinov2) → 正規化 → 位置チャンネル連結。
``dinov2`` はパッチの画素ではなく画像全体から作る (``app.services.deep``。``pca_dim`` は使わない)。
A・B で距離が比較可能になるよう、PCA 基底と zscore の統計は **A と B を結合して 1 組だけ**
求め、両者に同じ変換を適用する。

``zscore`` は各次元を標準化したあと ``1/sqrt(d)`` 倍する。二乗距離の平均が次元数に依らず約 2
(単位ベクトル同士と同じ) になるので、OT の ``blur`` を特徴の種類・次元から独立に選べる。
"""

from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor

from app.services.notices import Notice, WarningCode
from app.services.patches import GridMeta, patch_centers

FeatureType = Literal["raw", "pca", "color", "dinov2"]
Normalize = Literal["none", "zscore", "l2"]

_EPS = 1e-6
_PCA_SEED = 0
_PCA_NITER = 4


@dataclass(frozen=True)
class FeatureParams:
    type: FeatureType = "pca"
    pca_dim: int = 64
    normalize: Normalize = "zscore"
    position_weight: float = 0.0


@dataclass(frozen=True)
class Features:
    x: Tensor  # [N, d] float32, contiguous
    y: Tensor  # [M, d]
    warnings: list[Notice] = field(default_factory=list)

    @property
    def dim(self) -> int:
        return int(self.x.shape[1])


def build_features(
    patches_a: Tensor,
    grid_a: GridMeta,
    patches_b: Tensor,
    grid_b: GridMeta,
    params: FeatureParams,
    device: torch.device | str = "cpu",
    *,
    images: tuple[Tensor, Tensor] | None = None,
) -> Features:
    """2画像のパッチから ``X[N, d]``, ``Y[M, d]`` を作る。結果は ``device`` 上の float32。

    ``images`` は ``dinov2`` のときだけ必要な元画像 ``[3, H, W]`` ([0, 1]) の組 (A, B)。
    """
    _validate(patches_a, grid_a, patches_b, grid_b, params)
    warnings: list[Notice] = []

    n = patches_a.shape[0]

    if params.type == "dinov2":
        if images is None:
            raise ValueError("dinov2 features require the images")
        from app.services.deep import dinov2_features

        z = torch.cat([dinov2_features(images[0], grid_a, device),
                       dinov2_features(images[1], grid_b, device)])  # fmt: skip
    else:
        # dinov2 はパッチの画素を使わないので、転送はそれ以外のときだけ
        raw = torch.cat([patches_a, patches_b]).to(device=device, dtype=torch.float32)
        if params.type == "raw":
            z = raw
        elif params.type == "color":
            z = raw.view(-1, 3, grid_a.patch_size**2).mean(dim=-1)
        else:
            z, warning = _pca(raw, params.pca_dim)
            if warning:
                warnings.append(warning)

    z = _normalize(z, params.normalize)
    x, y = z[:n], z[n:]

    if params.position_weight > 0:
        x = _with_position(x, grid_a, params.position_weight)
        y = _with_position(y, grid_b, params.position_weight)

    return Features(x.contiguous(), y.contiguous(), warnings)


def _validate(
    patches_a: Tensor, grid_a: GridMeta, patches_b: Tensor, grid_b: GridMeta, p: FeatureParams
) -> None:
    if patches_a.shape[1] != patches_b.shape[1]:
        raise ValueError("patch dimensions of A and B differ (patch_size must match)")
    if patches_a.shape[0] != grid_a.n or patches_b.shape[0] != grid_b.n:
        raise ValueError("number of patches does not match the grid")
    if p.type == "color" and patches_a.shape[1] != 3 * grid_a.patch_size**2:
        raise ValueError("color features require RGB patches")
    if p.type == "pca" and p.pca_dim < 1:
        raise ValueError(f"pca_dim must be >= 1 (got {p.pca_dim})")
    if p.position_weight < 0:
        raise ValueError(f"position_weight must be >= 0 (got {p.position_weight})")


def _pca(z: Tensor, pca_dim: int) -> tuple[Tensor, Notice | None]:
    """中心化した ``z`` を上位 ``k`` 主成分へ射影する (``k`` は次元・サンプル数で頭打ち)。"""
    k = min(pca_dim, z.shape[0], z.shape[1])
    warning = None
    if k < pca_dim:
        warning = Notice(
            WarningCode.PCA_DIM_REDUCED,
            f"pca_dim was reduced from {pca_dim} to {k} (limited by patch count / raw dim).",
            {"requested": pca_dim, "actual": k},
        )
    zc = z - z.mean(dim=0, keepdim=True)
    # pca_lowrank は乱数を使う。結果を決定的にするため、シードを固定しグローバル RNG は汚さない
    cuda_devices = list(range(torch.cuda.device_count())) if z.is_cuda else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(_PCA_SEED)
        _, _, v = torch.pca_lowrank(zc, q=k, center=False, niter=_PCA_NITER)
    return zc @ v[:, :k], warning


def _normalize(z: Tensor, mode: Normalize) -> Tensor:
    if mode == "none":
        return z
    if mode == "zscore":
        mean = z.mean(dim=0, keepdim=True)
        std = z.std(dim=0, correction=0, keepdim=True)
        std = torch.where(std < _EPS, torch.ones_like(std), std)  # 定数次元は 0 のまま残す
        return (z - mean) / (std * z.shape[1] ** 0.5)
    return z / z.norm(dim=1, keepdim=True).clamp_min(_EPS)


def _with_position(f: Tensor, grid: GridMeta, weight: float) -> Tensor:
    centers = patch_centers(grid, device=f.device)
    scale = torch.tensor([grid.image_width, grid.image_height], device=f.device, dtype=f.dtype)
    return torch.cat([f, weight * centers / scale], dim=1)
