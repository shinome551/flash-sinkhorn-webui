import dataclasses
import io
import json
import logging
import time

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from PIL import Image

from app.api.deps import GpuLockDep, ImageStoreDep, SettingsDep
from app.config import Settings
from app.errors import AppError
from app.ot import SolveParams
from app.schemas.common import ErrorResponse
from app.schemas.match import (
    GridInfo,
    HardTarget,
    MatchRequest,
    MatchResponse,
    MatchWarning,
    PatchMatch,
    Target,
)
from app.schemas.match import MatchStats as MatchStatsSchema
from app.services.device import get_device_info
from app.services.features import FeatureParams
from app.services.image_store import ImageStore
from app.services.matching import MatchParams, MatchResult, match_images

logger = logging.getLogger(__name__)
router = APIRouter()


def _to_params(req: MatchRequest, settings: Settings) -> MatchParams:
    return MatchParams(
        patch_size=req.patch.size,
        stride=req.patch.stride,
        feature=FeatureParams(**req.feature.model_dump()),
        solve=SolveParams(
            **req.ot.model_dump(exclude={"backend"}), max_final_iters=settings.max_final_iters
        ),
        backend=req.ot.backend or settings.default_backend,
        top_k=req.output.top_k,
        min_weight=req.output.min_weight,
        compute_divergence=req.output.compute_divergence,
    )


def _load(store: ImageStore, image_id: str) -> Image.Image:
    img = Image.open(io.BytesIO(store.read_png(image_id)))
    img.load()
    return img.convert("RGB")


def _to_response(result: MatchResult) -> MatchResponse:
    plan, hard = result.plan, result.hard
    hard_j, hard_cost, hard_disp = (t.tolist() for t in (hard.idx, hard.cost, hard.displacement))
    idx, weight, cost, valid = (
        t.tolist() for t in (plan.top_idx, plan.top_weight, plan.top_cost, plan.top_valid)
    )
    matches = [
        PatchMatch(
            i=i,
            confidence=confidence,
            entropy=entropy,
            row_mass=row_mass,
            displacement=(dx, dy),
            targets=[
                Target(j=j, weight=w, cost=c)
                for j, w, c, ok in zip(idx[i], weight[i], cost[i], valid[i], strict=True)
                if ok
            ],
            hard=HardTarget(j=hard_j[i], cost=hard_cost[i], displacement=tuple(hard_disp[i])),
        )
        for i, (confidence, entropy, row_mass, (dx, dy)) in enumerate(
            zip(
                plan.confidence.tolist(),
                plan.entropy.tolist(),
                (
                    plan.row_mass * result.grid_a.n
                ).tolist(),  # a_i = 1/N で割る (col_mass と同じ規約)
                plan.displacement.tolist(),
                strict=True,
            )
        )
    ]
    return MatchResponse(
        grid_a=GridInfo(**dataclasses.asdict(result.grid_a)),
        grid_b=GridInfo(**dataclasses.asdict(result.grid_b)),
        matches=matches,
        col_mass=plan.col_mass.tolist(),
        hard_col_mass=hard.col_mass.tolist(),
        stats=MatchStatsSchema(**dataclasses.asdict(result.stats)),
        warnings=[MatchWarning(**dataclasses.asdict(w)) for w in result.warnings],
    )


def _log(req: MatchRequest, **fields: object) -> None:
    # 1 リクエスト 1 行の JSON (grep / jq で追える)
    record = {
        "image_a": req.image_a,
        "image_b": req.image_b,
        **req.model_dump(exclude={"image_a", "image_b"}),
        **fields,
    }
    logger.info("match %s", json.dumps(record, ensure_ascii=False))


@router.post(
    "/match",
    response_model=MatchResponse,
    responses={
        404: {"model": ErrorResponse, "description": "IMAGE_NOT_FOUND"},
        422: {"model": ErrorResponse, "description": "TOO_MANY_PATCHES / INVALID_PARAMS"},
        500: {"model": ErrorResponse, "description": "SOLVER_FAILED"},
    },
)
async def match(
    req: MatchRequest, store: ImageStoreDep, settings: SettingsDep, gpu_lock: GpuLockDep
) -> MatchResponse:
    started = time.perf_counter()
    try:
        img_a = await run_in_threadpool(_load, store, req.image_a)
        img_b = (
            img_a
            if req.image_b == req.image_a
            else await run_in_threadpool(_load, store, req.image_b)
        )
        params = _to_params(req, settings)
        limit = settings.patch_limit(get_device_info().cuda_available)
        # GPU は同時 1 件。待ち時間はソルバの所要時間に含めない (elapsed_ms は計算時間のみ)
        async with gpu_lock:
            result = await run_in_threadpool(
                match_images,
                img_a,
                img_b,
                params,
                max_patches=limit,
                warn_row_mass_error=settings.warn_row_mass_error,
            )
    except AppError as e:
        _log(req, status="error", code=str(e.code), message=e.message)
        raise
    _log(
        req,
        status="ok",
        backend=result.stats.backend,
        n_patches=[result.stats.n_patches_a, result.stats.n_patches_b],
        elapsed_ms=result.stats.elapsed_ms,
        request_ms=round((time.perf_counter() - started) * 1000, 2),
        warnings=len(result.warnings),
    )
    # N 件の Pydantic モデル生成は数十 ms かかるので、イベントループを塞がないようスレッドで
    return await run_in_threadpool(_to_response, result)
