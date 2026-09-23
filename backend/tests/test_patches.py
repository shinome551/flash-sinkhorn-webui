import pytest
import torch
from PIL import Image

from app.errors import AppError
from app.services.patches import (
    compute_grid,
    extract_patches,
    image_to_tensor,
    index_to_rowcol,
    patch_centers,
    patch_origin,
    rowcol_to_index,
)


def coded_image(c: int, h: int, w: int) -> torch.Tensor:
    """画素値が一意 (0 .. C*H*W-1) なので、どの画素がどこへ入ったかを厳密に追える。"""
    return torch.arange(c * h * w, dtype=torch.float32).reshape(c, h, w)


@pytest.mark.parametrize(
    ("w", "h", "size", "stride", "rows", "cols"),
    [
        (64, 48, 16, 16, 3, 4),
        (64, 48, 16, 8, 5, 7),  # 重なりあり
        (52, 50, 16, 16, 3, 3),  # 端数は捨てる
        (16, 16, 16, 16, 1, 1),  # 画像 = 1 パッチ
        (512, 384, 16, 16, 24, 32),  # SPEC 4.4 の例
    ],
)
def test_grid_shape(w, h, size, stride, rows, cols):
    grid = compute_grid(w, h, size, stride)
    assert (grid.rows, grid.cols, grid.n) == (rows, cols, rows * cols)
    assert (grid.image_width, grid.image_height) == (w, h)
    patches, grid2 = extract_patches(coded_image(3, h, w), size, stride)
    assert grid2 == grid
    assert patches.shape == (rows * cols, 3 * size * size)
    assert patches.is_contiguous()


def test_patch_pixels_match_source():
    c, h, w, size, stride = 3, 40, 52, 8, 4
    img = coded_image(c, h, w)
    patches, grid = extract_patches(img, size, stride)
    assert grid.n == 9 * 12
    for i in range(grid.n):
        x0, y0 = patch_origin(grid, i)
        expected = img[:, y0 : y0 + size, x0 : x0 + size].reshape(-1)  # チャンネル → 行 → 列
        assert torch.equal(patches[i], expected), f"patch {i} at ({x0}, {y0})"


def test_patch_centers():
    grid = compute_grid(64, 48, 16, 8)  # rows=5, cols=7
    centers = patch_centers(grid)
    assert centers.shape == (grid.n, 2) and centers.dtype == torch.float32
    assert centers[0].tolist() == [8.0, 8.0]
    assert centers[-1].tolist() == [6 * 8 + 8.0, 4 * 8 + 8.0]
    for i in range(grid.n):
        row, col = divmod(i, grid.cols)
        assert centers[i].tolist() == [col * 8 + 8.0, row * 8 + 8.0]
        x0, y0 = patch_origin(grid, i)
        assert centers[i].tolist() == [x0 + 8.0, y0 + 8.0]


def test_index_conversion_roundtrip():
    grid = compute_grid(64, 48, 16, 8)
    for i in range(grid.n):
        row, col = index_to_rowcol(i, grid.cols)
        assert 0 <= row < grid.rows and 0 <= col < grid.cols
        assert rowcol_to_index(row, col, grid.cols) == i
    assert index_to_rowcol(9, 7) == (1, 2)
    assert rowcol_to_index(1, 2, 7) == 9


def test_pil_input_is_scaled_to_unit_range():
    img = Image.new("RGB", (32, 16), (255, 0, 51))
    t = image_to_tensor(img)
    assert t.shape == (3, 16, 32) and t.dtype == torch.float32
    assert t[:, 0, 0].tolist() == pytest.approx([1.0, 0.0, 0.2])
    patches, grid = extract_patches(img, 16, 16)
    assert patches.shape == (2, 3 * 16 * 16) and grid.n == 2
    assert patches[0].view(3, -1)[:, 0].tolist() == pytest.approx([1.0, 0.0, 0.2])  # チャンネル順


def test_patch_larger_than_image_is_invalid_params():
    with pytest.raises(AppError) as exc:
        compute_grid(24, 64, 32, 32)
    assert exc.value.status_code == 422 and exc.value.code == "INVALID_PARAMS"


@pytest.mark.parametrize("size, stride", [(0, 16), (16, 0), (-1, 16)])
def test_non_positive_params_rejected(size, stride):
    with pytest.raises(ValueError):
        compute_grid(64, 64, size, stride)


def test_extraction_is_deterministic():
    img = coded_image(3, 32, 32)
    a, _ = extract_patches(img, 8, 8)
    b, _ = extract_patches(img, 8, 8)
    assert torch.equal(a, b)
