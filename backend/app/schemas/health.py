from pydantic import BaseModel


class HealthLimits(BaseModel):
    max_patches: int
    max_upload_bytes: int


class HealthResponse(BaseModel):
    status: str
    device: str
    cuda_available: bool
    gpu_name: str | None
    torch_version: str
    flash_sinkhorn_version: str | None
    default_backend: str
    feature_types: list[str]  # 利用できる feature.type (dinov2 は extra `deep` 導入時のみ)
    limits: HealthLimits
