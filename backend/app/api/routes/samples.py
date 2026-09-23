from pathlib import Path

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from app.api.deps import ImageStoreDep, SettingsDep
from app.api.routes.images import process_and_store
from app.config import Settings
from app.errors import AppError
from app.schemas.common import ErrorCode, ErrorResponse
from app.schemas.image import ImageUploadResponse
from app.schemas.sample import SampleInfo, SamplePairResponse
from app.services.image_store import ImageStore
from app.services.samples import load_samples

router = APIRouter()


def _store_file(path: Path, store: ImageStore, settings: Settings) -> ImageUploadResponse:
    """ファイルの読み込みも含めてスレッドで行う (イベントループを塞がない)。"""
    return process_and_store(
        path.read_bytes(),
        store,
        max_side=settings.max_image_side,
        resize_max=settings.resize_max,
        stride=settings.preprocess_stride,
    )


@router.get("/samples", response_model=list[SampleInfo])
def list_samples(settings: SettingsDep) -> list[SampleInfo]:
    return [
        SampleInfo(id=s.id, title=s.title, description=s.description)
        for s in load_samples(settings.samples_dir)
    ]


@router.post(
    "/samples/{sample_id}",
    response_model=SamplePairResponse,
    responses={404: {"model": ErrorResponse, "description": "HTTP_ERROR (unknown sample)"}},
)
async def load_sample(
    sample_id: str, store: ImageStoreDep, settings: SettingsDep
) -> SamplePairResponse:
    """サンプルの 2 枚を、アップロードと同じ前処理で画像ストアに登録する。"""
    sample = next((s for s in load_samples(settings.samples_dir) if s.id == sample_id), None)
    if sample is None:
        raise AppError(
            ErrorCode.HTTP_ERROR,
            f"Unknown sample: {sample_id}",
            status_code=404,
            hint="See GET /api/samples.",
        )
    a, b = [
        await run_in_threadpool(_store_file, path, store, settings)
        for path in (sample.path_a, sample.path_b)
    ]
    return SamplePairResponse(a=a, b=b)
