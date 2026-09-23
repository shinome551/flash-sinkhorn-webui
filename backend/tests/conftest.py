import io
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app


def make_image(width: int, height: int, mode: str = "RGB", seed: int = 0) -> Image.Image:
    """ノイズ画像 (PNG 圧縮が効かず、サイズ超過テストに使える)。"""
    rng = np.random.default_rng(seed)
    channels = len(Image.new(mode, (1, 1)).getbands())
    arr = rng.integers(0, 256, (height, width, channels), dtype=np.uint8)
    return Image.fromarray(arr.squeeze() if channels == 1 else arr, mode)


def encode(img: Image.Image, fmt: str = "PNG", **kwargs: object) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        tmp_dir=tmp_path, max_upload_bytes=200_000, warmup_on_startup=False, _env_file=None
    )  # type: ignore[call-arg]


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as c:
        yield c
