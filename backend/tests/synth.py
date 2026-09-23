"""OT のテスト用に、パッチごとに固有な内容を持つ合成テクスチャ画像を作る。"""

import torch
from torch.nn import functional as F

from app.services.features import FeatureParams, build_features
from app.services.patches import GridMeta, extract_patches, patch_centers


def texture(h: int, w: int, seed: int) -> torch.Tensor:
    """複数スケールの滑らかなノイズを重ねた ``[3, H, W]`` (0..1)。平坦なパッチが少なく対応が一意になる。"""
    g = torch.Generator().manual_seed(seed)
    img = torch.zeros(3, h, w)
    for k, s in enumerate((64, 32, 16, 8)):
        noise = torch.randn(3, 1, h // s + 2, w // s + 2, generator=g)
        img += F.interpolate(noise, size=(h, w), mode="bilinear", align_corners=False)[:, 0] / (
            1 + 0.5 * k
        )
    return (img - img.min()) / (img.max() - img.min())


def shifted_pair(
    h: int, w: int, dx: int, dy: int, seed: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """``B(p) = A(p + (dx, dy))`` となる 2 画像。A のパッチ内容は B の ``-(dx, dy)`` の位置にある。"""
    big = texture(h + dy, w + dx, seed)
    return big[:, :h, :w].contiguous(), big[:, dy : dy + h, dx : dx + w].contiguous()


def problem(
    img_a: torch.Tensor,
    img_b: torch.Tensor,
    patch: int = 8,
    device: str = "cpu",
    feature: FeatureParams | None = None,
) -> dict:
    """画像 2 枚 → ソルバ入力一式 (x, y, a, b, 中心座標, 格子)。重みは一様。"""
    pa, ga = extract_patches(img_a, patch, patch)
    pb, gb = extract_patches(img_b, patch, patch)
    feats = build_features(pa, ga, pb, gb, feature or FeatureParams(pca_dim=32), device)
    n, m = ga.n, gb.n
    return {
        "x": feats.x,
        "y": feats.y,
        "a": torch.full((n,), 1.0 / n, device=device),
        "b": torch.full((m,), 1.0 / m, device=device),
        "centers_a": patch_centers(ga, device),
        "centers_b": patch_centers(gb, device),
        "grid_a": ga,
        "grid_b": gb,
    }


def true_targets(ga: GridMeta, gb: GridMeta, dx: int, dy: int) -> torch.Tensor:
    """``shifted_pair`` の真の対応 ``[N]`` (B 側 index、B の外に出る A パッチは -1)。dx, dy は stride の倍数。"""
    s = ga.stride
    out = torch.full((ga.n,), -1, dtype=torch.long)
    for r in range(ga.rows):
        for c in range(ga.cols):
            rr, cc = r - dy // s, c - dx // s
            if 0 <= rr < gb.rows and 0 <= cc < gb.cols:
                out[r * ga.cols + c] = rr * gb.cols + cc
    return out
