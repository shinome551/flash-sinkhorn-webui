from typing import Literal

from app.ot.base import BackendUnavailableError, SinkhornBackend, SolveInfo, SolveParams
from app.ot.dense_backend import DenseBackend
from app.services.device import resolve_backend

__all__ = [
    "BackendUnavailableError",
    "DenseBackend",
    "SinkhornBackend",
    "SolveInfo",
    "SolveParams",
    "get_backend",
]

_instances: dict[str, SinkhornBackend] = {}


def get_backend(setting: Literal["auto", "flash", "dense"]) -> SinkhornBackend:
    """``auto`` は CUDA と flash_sinkhorn が使えれば flash、なければ dense。

    ``flash`` を明示して使えない場合は ``BackendUnavailableError`` (フォールバックしない)。
    """
    name = resolve_backend(setting)
    if name not in _instances:
        if name == "flash":
            from app.ot.flash_backend import FlashBackend

            _instances[name] = FlashBackend()
        else:
            _instances[name] = DenseBackend()
    return _instances[name]
