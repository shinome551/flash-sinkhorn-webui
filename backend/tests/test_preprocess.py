import io

import pytest
from PIL import Image

from app.errors import AppError
from app.services.preprocess import encode_png, preprocess_upload
from tests.conftest import encode, make_image

KW = {"max_side": 8192, "resize_max": 512, "stride": 16}


def test_exif_orientation_is_applied():
    # 左半分が赤・右半分が青の 64x32。Orientation=6 (時計回り 90°) で表示すると上=赤・下=青の 32x64
    img = Image.new("RGB", (64, 32))
    img.paste((255, 0, 0), (0, 0, 32, 32))
    img.paste((0, 0, 255), (32, 0, 64, 32))
    exif = Image.Exif()
    exif[0x0112] = 6
    result = preprocess_upload(encode(img, "JPEG", exif=exif, quality=100), **KW)
    out = result.image
    assert out.size == (32, 64)
    assert (result.original_width, result.original_height) == (32, 64)
    r, _, b = out.getpixel((16, 8))
    assert r > 200 and b < 60  # 上側が赤
    r, _, b = out.getpixel((16, 56))
    assert b > 200 and r < 60  # 下側が青


def test_alpha_is_composited_on_white():
    rgba = Image.new("RGBA", (32, 32), (255, 0, 0, 0))  # 完全透明の赤
    rgba.paste((0, 0, 0, 255), (0, 0, 16, 32))  # 左半分は不透明な黒
    out = preprocess_upload(encode(rgba), **KW).image
    assert out.mode == "RGB"
    assert out.getpixel((30, 10)) == (255, 255, 255)
    assert out.getpixel((2, 10)) == (0, 0, 0)


def test_palette_with_transparency_is_composited():
    p = Image.new("P", (32, 32), 0)
    p.putpalette([255, 0, 0] + [0, 0, 0] * 255)
    p.info["transparency"] = 0
    out = preprocess_upload(encode(p, transparency=0), **KW).image
    assert out.getpixel((5, 5)) == (255, 255, 255)


@pytest.mark.parametrize("mode", ["L", "LA", "P", "CMYK"])
def test_other_modes_become_rgb(mode: str):
    img = make_image(32, 32, mode) if mode != "P" else make_image(32, 32).convert("P")
    fmt = "JPEG" if mode == "CMYK" else "PNG"
    assert preprocess_upload(encode(img, fmt), **KW).image.mode == "RGB"


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ((512, 512), (512, 512)),  # 変更なし
        ((100, 50), (96, 48)),  # 縮小せず切り詰めのみ
        ((2000, 1000), (512, 256)),  # 縮小のみ
        ((1000, 600), (512, 304)),  # 縮小 + 切り詰め
        ((300, 1200), (128, 512)),  # 縦長: 300*512/1200 = 128
    ],
)
def test_resize_and_trim(size: tuple[int, int], expected: tuple[int, int]):
    assert preprocess_upload(encode(make_image(*size)), **KW).image.size == expected


def test_trim_keeps_top_left_pixels():
    img = make_image(40, 40)
    out = preprocess_upload(encode(img), **KW).image
    assert out.size == (32, 32)
    assert out.tobytes() == img.crop((0, 0, 32, 32)).tobytes()


def test_too_small_after_resize_rejected():
    with pytest.raises(AppError) as exc:
        preprocess_upload(encode(make_image(15, 40)), **KW)
    assert exc.value.status_code == 422


def test_encode_png_roundtrip():
    img = make_image(32, 32)
    assert Image.open(io.BytesIO(encode_png(img))).tobytes() == img.tobytes()
