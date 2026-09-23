from pathlib import Path

import pytest

from app.errors import AppError
from app.services.image_store import (
    ImageStore,
    is_valid_image_id,
    new_ulid,
    ulid_timestamp_ms,
)


class Clock:
    def __init__(self, now: float = 1_800_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_ulid_format_and_timestamp():
    ulid = new_ulid(1_800_000_000_123)
    assert len(ulid) == 26 and is_valid_image_id(ulid)
    assert ulid_timestamp_ms(ulid) == 1_800_000_000_123
    assert new_ulid() != new_ulid()


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "..",
        "../" + "0" * 23,
        "0" * 25,  # 短い
        "0" * 27,  # 長い
        "8" + "0" * 25,  # 先頭は 0-7 (128bit に収まらない)
        "0" * 25 + "U",  # Crockford に無い文字
        "0" * 25 + "i",  # 小文字は不可
        "0" * 25 + "\n",  # $ 落ち対策
        "0" * 25 + "/",
    ],
)
def test_invalid_ids_rejected(bad: str):
    assert not is_valid_image_id(bad)


def test_save_and_read(tmp_path: Path):
    store = ImageStore(tmp_path, ttl_seconds=60, clock=Clock())
    image_id = store.save(b"png-bytes")
    assert store.read_png(image_id) == b"png-bytes"
    assert [p.name for p in tmp_path.iterdir()] == [f"{image_id}.png"]  # .tmp が残らない


def test_read_invalid_or_unknown_id_is_not_found(tmp_path: Path):
    (tmp_path.parent / "secret.png").write_bytes(b"secret")
    store = ImageStore(tmp_path, ttl_seconds=60, clock=Clock())
    for bad in ["../secret", "..", "x", new_ulid()]:
        with pytest.raises(AppError) as exc:
            store.read_png(bad)
        assert exc.value.status_code == 404


def test_expired_image_is_not_found_and_swept(tmp_path: Path):
    clock = Clock()
    store = ImageStore(tmp_path, ttl_seconds=60, clock=clock)
    image_id = store.save(b"x")
    clock.now += 59
    assert store.read_png(image_id) == b"x"
    clock.now += 2
    with pytest.raises(AppError):
        store.read_png(image_id)
    assert store.sweep() == 1
    assert list(tmp_path.iterdir()) == []


def test_sweep_keeps_fresh_and_ignores_foreign_files(tmp_path: Path):
    clock = Clock()
    store = ImageStore(tmp_path, ttl_seconds=60, clock=clock)
    old = store.save(b"old")
    (tmp_path / f"{new_ulid(int(clock.now * 1000))}.png.tmp").write_bytes(b"stale tmp")
    (tmp_path / "README.txt").write_text("not ours")
    clock.now += 100
    fresh = store.save(b"fresh")  # save が期限切れ掃除も起動する
    names = {p.name for p in tmp_path.iterdir()}
    assert f"{old}.png" not in names
    assert f"{fresh}.png" in names and "README.txt" in names
    assert not any(n.endswith(".tmp") for n in names)
