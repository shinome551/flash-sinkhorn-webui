"""実行デバイスと OT バックエンド可用性の判定。"""

import importlib.metadata
import importlib.util
from dataclasses import dataclass
from functools import cache
from typing import Literal

import torch


@dataclass(frozen=True)
class DeviceInfo:
    device: str
    cuda_available: bool
    gpu_name: str | None
    torch_version: str
    flash_sinkhorn_version: str | None


@cache
def get_device_info() -> DeviceInfo:
    cuda = torch.cuda.is_available()
    try:
        flash_version: str | None = importlib.metadata.version("flash-sinkhorn")
    except importlib.metadata.PackageNotFoundError:
        flash_version = None
    return DeviceInfo(
        device="cuda:0" if cuda else "cpu",
        cuda_available=cuda,
        gpu_name=torch.cuda.get_device_name(0) if cuda else None,
        torch_version=torch.__version__,
        flash_sinkhorn_version=flash_version,
    )


def resolve_backend(setting: Literal["auto", "flash", "dense"]) -> Literal["flash", "dense"]:
    """``auto`` は CUDA と flash_sinkhorn が使えるときだけ flash、それ以外は dense。"""
    if setting != "auto":
        return setting
    usable = get_device_info().cuda_available and importlib.util.find_spec("flash_sinkhorn")
    return "flash" if usable else "dense"
