import asyncio
from typing import Annotated

from fastapi import Depends, Request

from app.config import Settings
from app.services.image_store import ImageStore


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_image_store(request: Request) -> ImageStore:
    store: ImageStore = request.app.state.image_store
    return store


def get_gpu_lock(request: Request) -> asyncio.Semaphore:
    lock: asyncio.Semaphore = request.app.state.gpu_lock
    return lock


SettingsDep = Annotated[Settings, Depends(get_settings)]
GpuLockDep = Annotated[asyncio.Semaphore, Depends(get_gpu_lock)]
ImageStoreDep = Annotated[ImageStore, Depends(get_image_store)]
