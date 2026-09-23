"""深層特徴 (DINOv2 ViT-S/14, 384 次元)。optional extra ``deep`` (``uv sync --extra deep``) が必要。

トークン格子をパッチ格子に合わせるため、**トークン間隔 = stride** になるよう入力を作る:
格子が覆う領域 ``[0, cols*stride) x [0, rows*stride)`` を ``(cols*14, rows*14)`` にリサイズして
モデルに通すと、トークン ``(r, c)`` は元画像の ``[c*stride, (c+1)*stride)`` を代表する。
``patch_size == stride`` ならトークン 1 個 = パッチ 1 個で一致し、そうでなければトークンマップを
パッチ中心で双線形補間する。パッチサイズを変えるとモデルが見る解像度も変わる点に注意。

モデルは初回呼び出しで読み込む (重みは Hugging Face Hub から取得してキャッシュ)。
"""

import importlib.util
import threading
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from app.services.patches import GridMeta

MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"
DIM = 384
TOKEN_SIZE = 14

_models: dict[str, Any] = {}
_lock = threading.Lock()


def deep_available() -> bool:
    """``timm`` が導入済みか (モデルの重みの取得可否までは見ない)。"""
    return importlib.util.find_spec("timm") is not None


class ModelLoadError(RuntimeError):
    """モデルの生成・重みの取得に失敗した (未導入、オフライン、キャッシュ破損など)。"""


def _get_model(device: torch.device) -> Any:
    key = str(device)
    with _lock:
        if key not in _models:
            try:
                import timm

                model = timm.create_model(MODEL_NAME, pretrained=True, dynamic_img_size=True)
            except Exception as e:
                raise ModelLoadError(f"{type(e).__name__}: {e}") from e
            _models[key] = model.eval().to(device)
        return _models[key]


def dinov2_features(img: Tensor, grid: GridMeta, device: torch.device | str = "cpu") -> Tensor:
    """``[3, H, W]`` ([0, 1]) の画像から、格子の各パッチの特徴 ``[N, 384]`` (float32) を作る。"""
    device = torch.device(device)
    model = _get_model(device)
    s = grid.stride
    region_w, region_h = grid.cols * s, grid.rows * s

    x = img.to(device=device, dtype=torch.float32).unsqueeze(0)
    # stride > patch_size だと領域が画像をはみ出すので端を複製して埋める
    pad_w, pad_h = max(region_w - x.shape[-1], 0), max(region_h - x.shape[-2], 0)
    if pad_w or pad_h:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    x = x[..., :region_h, :region_w]
    x = F.interpolate(
        x, size=(grid.rows * TOKEN_SIZE, grid.cols * TOKEN_SIZE),
        mode="bilinear", align_corners=False, antialias=True,
    )  # fmt: skip
    cfg = model.pretrained_cfg
    mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)

    with torch.no_grad():
        tokens = model.forward_features((x - mean) / std)[:, model.num_prefix_tokens :]
    tmap = tokens.reshape(1, grid.rows, grid.cols, DIM).permute(0, 3, 1, 2)  # [1, D, rows, cols]

    if grid.patch_size == s:
        feats = tmap
    else:
        # パッチ中心 (元画像の px) → トークン格子上の正規化座標 (align_corners=False: 端が ±1)
        half = grid.patch_size / 2
        u = (torch.arange(grid.cols, device=device) * s + half) / region_w * 2 - 1
        v = (torch.arange(grid.rows, device=device) * s + half) / region_h * 2 - 1
        gv, gu = torch.meshgrid(v, u, indexing="ij")
        sample_at = torch.stack([gu, gv], dim=-1).unsqueeze(0)  # [1, rows, cols, 2] = (x, y)
        feats = F.grid_sample(
            tmap, sample_at, mode="bilinear", padding_mode="border", align_corners=False
        )
    return feats.squeeze(0).flatten(1).t().contiguous().float()
