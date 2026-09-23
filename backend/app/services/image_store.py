"""アップロード画像の一時ストア。

``{root}/{image_id}.png`` の1ファイル構成。``image_id`` は ULID で、先頭 48bit の
ミリ秒タイムスタンプを TTL 判定にも使う (メタデータファイルは持たない)。
外部入力の ``image_id`` は必ず ``validate_image_id`` を通してからパスに使う。
"""

import os
import re
import time
from collections.abc import Callable
from pathlib import Path

from app.errors import image_not_found

_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford base32
_ULID_RE = re.compile(r"[0-7][0-9A-HJKMNP-TV-Z]{25}")
_SWEEP_INTERVAL_S = 60.0


def new_ulid(now_ms: int | None = None) -> str:
    ts = int(time.time() * 1000) if now_ms is None else now_ms
    value = (ts << 80) | int.from_bytes(os.urandom(10), "big")
    chars = []
    for _ in range(26):
        chars.append(_ALPHABET[value & 31])
        value >>= 5
    return "".join(reversed(chars))


def is_valid_image_id(image_id: str) -> bool:
    # fullmatch: ``$`` は末尾の改行にもマッチするため使わない
    return _ULID_RE.fullmatch(image_id) is not None


def ulid_timestamp_ms(image_id: str) -> int:
    value = 0
    for ch in image_id:
        value = (value << 5) | _ALPHABET.index(ch)
    return value >> 80


class ImageStore:
    def __init__(
        self, root: Path, ttl_seconds: float, clock: Callable[[], float] = time.time
    ) -> None:
        self.root = root
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._last_sweep = 0.0
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, png: bytes) -> str:
        self.sweep_if_due()
        image_id = new_ulid(int(self._clock() * 1000))
        tmp = self.root / f"{image_id}.png.tmp"
        tmp.write_bytes(png)
        os.replace(tmp, self._path(image_id))  # 書き込み途中のファイルを読ませない
        return image_id

    def read_png(self, image_id: str) -> bytes:
        """PNG のバイト列を返す。不正 ID・不明・期限切れはすべて IMAGE_NOT_FOUND。"""
        if not is_valid_image_id(image_id) or self._expired(image_id):
            raise image_not_found(image_id)
        try:
            return self._path(image_id).read_bytes()
        except FileNotFoundError:
            raise image_not_found(image_id) from None

    def sweep(self) -> int:
        """期限切れ (と取り残された ``.tmp``) を削除し、削除件数を返す。"""
        self._last_sweep = self._clock()
        removed = 0
        for path in self.root.iterdir():
            image_id = path.name.split(".", 1)[0]
            if not is_valid_image_id(image_id) or not self._expired(image_id):
                continue
            try:
                path.unlink()
                removed += 1
            except FileNotFoundError:
                pass  # 並行する sweep が先に消した
        return removed

    def sweep_if_due(self) -> None:
        if self._clock() - self._last_sweep >= _SWEEP_INTERVAL_S:
            self.sweep()

    def _path(self, image_id: str) -> Path:
        return self.root / f"{image_id}.png"

    def _expired(self, image_id: str) -> bool:
        age = self._clock() - ulid_timestamp_ms(image_id) / 1000
        return age >= self.ttl_seconds
