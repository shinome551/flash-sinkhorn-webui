from fastapi import APIRouter, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from app.api.deps import ImageStoreDep, SettingsDep
from app.errors import AppError
from app.schemas.common import ErrorCode, ErrorResponse
from app.schemas.image import ImageUploadResponse
from app.services.image_store import ImageStore
from app.services.preprocess import encode_png, preprocess_upload

router = APIRouter()


def process_and_store(
    data: bytes, store: ImageStore, *, max_side: int, resize_max: int, stride: int
) -> ImageUploadResponse:
    result = preprocess_upload(data, max_side=max_side, resize_max=resize_max, stride=stride)
    image_id = store.save(encode_png(result.image))
    return ImageUploadResponse(
        image_id=image_id,
        width=result.image.width,
        height=result.image.height,
        original_width=result.original_width,
        original_height=result.original_height,
        url=f"/api/images/{image_id}",
    )


@router.post(
    "/images",
    response_model=ImageUploadResponse,
    responses={
        413: {"model": ErrorResponse, "description": "IMAGE_TOO_LARGE"},
        415: {"model": ErrorResponse, "description": "UNSUPPORTED_FORMAT"},
        422: {"model": ErrorResponse, "description": "INVALID_PARAMS"},
    },
)
async def upload_image(
    file: UploadFile, store: ImageStoreDep, settings: SettingsDep
) -> ImageUploadResponse:
    # 上限+1 バイトだけ読んで超過を検知する (巨大ボディを丸ごとメモリに載せない)
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise AppError(
            ErrorCode.IMAGE_TOO_LARGE,
            f"Upload exceeds {settings.max_upload_bytes} bytes.",
            status_code=413,
            hint="Use a smaller file.",
        )
    return await run_in_threadpool(
        process_and_store,
        data,
        store,
        max_side=settings.max_image_side,
        resize_max=settings.resize_max,
        stride=settings.preprocess_stride,
    )


@router.get(
    "/images/{image_id}",
    response_class=Response,
    responses={
        200: {"content": {"image/png": {}}},
        404: {"model": ErrorResponse, "description": "IMAGE_NOT_FOUND"},
    },
)
def get_image(image_id: str, store: ImageStoreDep) -> Response:
    # bytes で読み切ってから返す (FileResponse だと配信中に TTL 掃除と競合しうる)
    return Response(content=store.read_png(image_id), media_type="image/png")
