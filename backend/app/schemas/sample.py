from pydantic import BaseModel

from app.schemas.image import ImageUploadResponse


class SampleInfo(BaseModel):
    id: str
    title: str
    description: str


class SamplePairResponse(BaseModel):
    """``POST /api/samples/{id}``: サンプルの 2 枚を画像ストアに登録した結果 (アップロードと同じ形)。"""

    a: ImageUploadResponse
    b: ImageUploadResponse
