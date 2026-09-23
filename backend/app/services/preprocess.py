"""アップロード画像の前処理 (SPEC 3.1)。

デコード → EXIF 回転 → RGB 化 (アルファは白背景に合成) → 長辺 ``resize_max`` へ縮小
→ 幅・高さを ``stride`` の倍数に切り詰め (右端・下端を削る)。
"""

import io
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

from app.errors import AppError
from app.schemas.common import ErrorCode

ALLOWED_FORMATS = frozenset({"PNG", "JPEG", "WEBP"})


@dataclass(frozen=True)
class PreprocessedImage:
    image: Image.Image  # RGB
    original_width: int  # EXIF 回転後・縮小前
    original_height: int


def preprocess_upload(
    data: bytes, *, max_side: int, resize_max: int, stride: int
) -> PreprocessedImage:
    """アップロードされたバイト列を前処理済み RGB 画像にする。拒否時は ``AppError``。"""
    img = _decode(data, max_side=max_side)
    img = ImageOps.exif_transpose(img)
    original_width, original_height = img.size
    img = _to_rgb(img)
    img = _resize_max(img, resize_max)
    img = _trim_to_stride(img, stride)
    return PreprocessedImage(img, original_width, original_height)


def encode_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=3)
    return buf.getvalue()


def _decode(data: bytes, *, max_side: int) -> Image.Image:
    unsupported = AppError(
        ErrorCode.UNSUPPORTED_FORMAT,
        "The file could not be decoded as an image.",
        status_code=415,
        hint="Use a PNG, JPEG or WebP image.",
    )
    try:
        img = Image.open(io.BytesIO(data))
        if img.format not in ALLOWED_FORMATS:
            raise unsupported
        # open() はヘッダしか読まない。巨大画像はピクセル展開の前に弾く
        if max(img.size) > max_side:
            raise _too_large(f"Image side exceeds {max_side}px: {img.width}x{img.height}.")
        img.load()
    except Image.DecompressionBombError:
        raise _too_large("Image has too many pixels.") from None
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise unsupported from None
    return img


def _too_large(message: str) -> AppError:
    return AppError(
        ErrorCode.IMAGE_TOO_LARGE, message, status_code=413, hint="Use a smaller image."
    )


def _to_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "I" or img.mode.startswith("I;16"):
        # 16 bit グレースケール。convert("RGB") は 255 で飽和してほぼ真っ白になるので上位 8 bit を取る
        img = img.convert("I").point(lambda v: v / 256).convert("L")
    if img.has_transparency_data:
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return img.convert("RGB")


def _resize_max(img: Image.Image, resize_max: int) -> Image.Image:
    width, height = img.size
    scale = resize_max / max(width, height)
    if scale >= 1:
        return img  # 拡大はしない
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def _trim_to_stride(img: Image.Image, stride: int) -> Image.Image:
    width, height = img.size
    new_width, new_height = width // stride * stride, height // stride * stride
    if new_width == 0 or new_height == 0:
        raise AppError(
            ErrorCode.INVALID_PARAMS,
            f"Image is too small: {width}x{height} after resizing (both sides must be >= {stride}px).",
            status_code=422,
            hint="Use a larger image.",
        )
    if (new_width, new_height) == (width, height):
        return img
    return img.crop((0, 0, new_width, new_height))
