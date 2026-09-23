from pydantic import BaseModel


class ImageUploadResponse(BaseModel):
    """``POST /api/images`` のレスポンス。width/height は前処理後、original_* は EXIF 回転後・縮小前。"""

    image_id: str
    width: int
    height: int
    original_width: int
    original_height: int
    url: str
