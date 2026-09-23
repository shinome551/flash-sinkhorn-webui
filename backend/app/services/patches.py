"""パッチ抽出と格子の座標変換 (SPEC 3.2)。

パッチ ``i`` は行優先で並ぶ: ``i = row * cols + col``。
座標系はピクセル端基準 (ピクセル ``k`` は ``[k, k+1)`` を占める) で、canvas の描画座標と一致する。
パッチ ``(row, col)`` の左上は ``(col * stride, row * stride)``、中心は左上 + ``patch_size / 2``。
"""

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F

from app.errors import AppError
from app.schemas.common import ErrorCode


@dataclass(frozen=True)
class GridMeta:
    rows: int
    cols: int
    patch_size: int
    stride: int
    image_width: int
    image_height: int

    @property
    def n(self) -> int:
        return self.rows * self.cols


def compute_grid(image_width: int, image_height: int, size: int, stride: int) -> GridMeta:
    """画素を読まずに格子の寸法だけを求める (パッチ数の上限チェック用)。"""
    if size <= 0 or stride <= 0:
        raise ValueError(f"size and stride must be positive (got {size}, {stride})")
    if image_width < size or image_height < size:
        raise AppError(
            ErrorCode.INVALID_PARAMS,
            f"Patch size {size} exceeds the image ({image_width}x{image_height}).",
            status_code=422,
            hint="Use a smaller patch size or a larger image.",
        )
    return GridMeta(
        rows=(image_height - size) // stride + 1,
        cols=(image_width - size) // stride + 1,
        patch_size=size,
        stride=stride,
        image_width=image_width,
        image_height=image_height,
    )


def image_to_tensor(img: Image.Image) -> Tensor:
    """PIL 画像 → ``[3, H, W]`` float32 ([0, 1])。"""
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(arr.copy()).permute(2, 0, 1).float().div_(255.0)


def extract_patches(img: Image.Image | Tensor, size: int, stride: int) -> tuple[Tensor, GridMeta]:
    """画像を格子状のパッチに切り出す。

    ``img`` は PIL 画像または ``[C, H, W]`` テンソル。戻り値の ``patches`` は
    ``[N, C*size*size]`` (次元の並びは チャンネル → 行 → 列) で、入力と同じデバイスに置かれる。
    """
    tensor = image_to_tensor(img) if isinstance(img, Image.Image) else img
    if tensor.ndim != 3:
        raise ValueError(f"expected a [C, H, W] tensor, got shape {tuple(tensor.shape)}")
    _, height, width = tensor.shape
    grid = compute_grid(width, height, size, stride)
    unfolded = F.unfold(tensor.unsqueeze(0), kernel_size=size, stride=stride)  # [1, C*P*P, N]
    return unfolded.squeeze(0).t().contiguous(), grid


def patch_centers(grid: GridMeta, device: torch.device | str = "cpu") -> Tensor:
    """各パッチ中心のピクセル座標 ``[N, 2]`` = ``(cx, cy)`` (float32)。"""
    rows = torch.arange(grid.rows, device=device, dtype=torch.float32)
    cols = torch.arange(grid.cols, device=device, dtype=torch.float32)
    half = grid.patch_size / 2
    cy = (rows * grid.stride + half).unsqueeze(1).expand(grid.rows, grid.cols)
    cx = (cols * grid.stride + half).unsqueeze(0).expand(grid.rows, grid.cols)
    return torch.stack([cx, cy], dim=-1).reshape(grid.n, 2)


def index_to_rowcol(i: int, cols: int) -> tuple[int, int]:
    return divmod(i, cols)


def rowcol_to_index(row: int, col: int, cols: int) -> int:
    return row * cols + col


def patch_origin(grid: GridMeta, i: int) -> tuple[int, int]:
    """パッチ ``i`` の左上ピクセル座標 ``(x0, y0)``。"""
    row, col = index_to_rowcol(i, grid.cols)
    return col * grid.stride, row * grid.stride
