from fastapi import APIRouter

from app.api.deps import SettingsDep
from app.schemas.health import HealthLimits, HealthResponse
from app.services.deep import deep_available
from app.services.device import get_device_info, resolve_backend

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(settings: SettingsDep) -> HealthResponse:
    info = get_device_info()
    return HealthResponse(
        status="ok",
        device=info.device,
        cuda_available=info.cuda_available,
        gpu_name=info.gpu_name,
        torch_version=info.torch_version,
        flash_sinkhorn_version=info.flash_sinkhorn_version,
        default_backend=resolve_backend(settings.default_backend),
        feature_types=["pca", "raw", "color"] + (["dinov2"] if deep_available() else []),
        limits=HealthLimits(
            max_patches=settings.patch_limit(info.cuda_available),
            max_upload_bytes=settings.max_upload_bytes,
        ),
    )
