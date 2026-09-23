import pytest
import torch

from app.services.features import FeatureParams, build_features
from app.services.patches import extract_patches, patch_centers

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"))]  # fmt: skip


def random_image(h: int, w: int, seed: int) -> torch.Tensor:
    return torch.rand(3, h, w, generator=torch.Generator().manual_seed(seed))


@pytest.fixture
def pair():
    """64x64 (patch 4 -> N=256, raw dim 48) の 2 画像。"""
    pa, ga = extract_patches(random_image(64, 64, 1), 4, 4)
    pb, gb = extract_patches(random_image(64, 64, 2), 4, 4)
    return pa, ga, pb, gb


def build(pair, device="cpu", **kw):
    pa, ga, pb, gb = pair
    return build_features(pa, ga, pb, gb, FeatureParams(**kw), device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    ("kw", "dim"),
    [
        ({"type": "raw"}, 48),
        ({"type": "color"}, 3),
        ({"type": "pca", "pca_dim": 16}, 16),
        ({"type": "pca", "pca_dim": 16, "position_weight": 0.5}, 18),
        ({"type": "raw", "position_weight": 1.0}, 50),
    ],
)
def test_dims_dtype_layout(pair, device, kw, dim):
    f = build(pair, device, **kw)
    assert f.x.shape == (256, dim) and f.y.shape == (256, dim) and f.dim == dim
    for t in (f.x, f.y):
        assert t.dtype == torch.float32 and t.is_contiguous()
        assert t.device.type == device
    assert f.warnings == []


def test_raw_without_normalize_is_identity(pair):
    f = build(pair, type="raw", normalize="none")
    assert torch.equal(f.x, pair[0]) and torch.equal(f.y, pair[2])


def test_color_is_mean_rgb():
    img = torch.zeros(3, 8, 16)
    img[0, :, :8], img[1, :, 8:] = 1.0, 0.5
    img[2, :4, :8] = 0.8  # 左パッチ青チャンネルの上半分だけ
    pa, ga = extract_patches(img, 8, 8)
    f = build_features(pa, ga, pa, ga, FeatureParams(type="color", normalize="none"))
    assert f.x[0].tolist() == pytest.approx([1.0, 0.0, 0.4])
    assert f.x[1].tolist() == pytest.approx([0.0, 0.5, 0.0])


@pytest.mark.parametrize("kind", ["raw", "color", "pca"])
def test_zscore_statistics_are_joint(pair, kind):
    f = build(pair, type=kind, pca_dim=16, normalize="zscore")
    z = torch.cat([f.x, f.y])
    assert torch.allclose(z.mean(0), torch.zeros(z.shape[1]), atol=1e-4)
    # 標準偏差は 1/sqrt(d) (二乗距離の平均を次元に依らず約 2 にするため)
    expected_std = torch.full((z.shape[1],), z.shape[1] ** -0.5)
    assert torch.allclose(z.std(0, correction=0), expected_std, atol=1e-4)


@pytest.mark.parametrize("kind", ["raw", "color", "pca"])
def test_zscore_mean_squared_distance_is_about_two(pair, kind):
    f = build(pair, type=kind, pca_dim=16, normalize="zscore")
    assert (torch.cdist(f.x, f.y) ** 2).mean().item() == pytest.approx(2.0, rel=0.1)


def test_zscore_keeps_difference_between_images():
    """統計は A・B 結合で 1 組。画像ごとに正規化すると A と B の平均差 (色の違い) が消える。"""
    pa, ga = extract_patches(random_image(64, 64, 1) * 0.5, 4, 4)  # 暗い画像
    pb, gb = extract_patches(random_image(64, 64, 2) * 0.5 + 0.4, 4, 4)  # 明るい画像
    f = build_features(pa, ga, pb, gb, FeatureParams(type="color", normalize="zscore"))
    assert torch.all(f.x.mean(0) < -0.5 / 3**0.5) and torch.all(f.y.mean(0) > 0.5 / 3**0.5)


def test_zscore_constant_dimension_stays_finite():
    img = torch.full((3, 16, 16), 0.5)
    img[0, :, :8] = 0.0  # R だけ変化する。G/B は全パッチ・全画素で定数
    pa, ga = extract_patches(img, 8, 8)
    f = build_features(pa, ga, pa, ga, FeatureParams(type="color", normalize="zscore"))
    assert torch.isfinite(f.x).all()
    assert torch.all(f.x[:, 1:] == 0)


def test_l2_norms_are_one(pair):
    f = build(pair, type="pca", pca_dim=16, normalize="l2")
    assert torch.allclose(f.x.norm(dim=1), torch.ones(256), atol=1e-5)
    assert torch.allclose(f.y.norm(dim=1), torch.ones(256), atol=1e-5)


def test_l2_zero_vector_stays_finite():
    zero = torch.zeros(3, 8, 8)
    pa, ga = extract_patches(zero, 8, 8)
    f = build_features(pa, ga, pa, ga, FeatureParams(type="raw", normalize="l2"))
    assert torch.isfinite(f.x).all()


def test_pca_shares_mean_and_basis_between_images():
    """B = A + 定数オフセットなら、共通の中心化により A-B の対応パッチ間距離がオフセット長に保たれる
    (A・B を別々に中心化・別基底にすると、この差が消えるか歪む)。"""
    pa, ga = extract_patches(random_image(64, 64, 3), 4, 4)  # raw 次元 48
    pb = pa + 0.3
    f = build_features(pa, ga, pb, ga, FeatureParams(type="pca", pca_dim=48, normalize="none"))
    expected = 0.3 * 48**0.5
    assert torch.allclose((f.y - f.x).norm(dim=1), torch.full((256,), expected), atol=1e-3)


def test_full_rank_pca_preserves_distances(pair):
    """全次元 PCA は共通の中心化 + 直交変換なので、A-B 間の距離が raw と一致する。"""
    raw = build(pair, type="raw", normalize="none")
    pca = build(pair, type="pca", pca_dim=48, normalize="none")
    assert torch.allclose(torch.cdist(pca.x, pca.y), torch.cdist(raw.x, raw.y), atol=1e-3)


def test_pca_captures_dominant_directions():
    """A・B が 2 つの主方向だけで張られるなら、pca_dim=2 で距離が保存される。"""
    g = torch.Generator().manual_seed(0)
    basis = torch.linalg.qr(torch.randn(48, 2, generator=g))[0]  # [48, 2]
    pa = torch.randn(200, 2, generator=g) @ basis.T * 5
    pb = torch.randn(200, 2, generator=g) @ basis.T * 5
    from app.services.patches import GridMeta

    ga = GridMeta(rows=10, cols=20, patch_size=4, stride=4, image_width=80, image_height=40)
    f = build_features(pa, ga, pb, ga, FeatureParams(type="pca", pca_dim=2, normalize="none"))
    assert torch.allclose(torch.cdist(f.x, f.y), torch.cdist(pa, pb), atol=1e-3)


@pytest.mark.parametrize("device", DEVICES)
def test_deterministic_and_leaves_global_rng_untouched(pair, device):
    torch.manual_seed(123)
    expected_next = torch.rand(1)
    torch.manual_seed(123)
    a = build(pair, device, type="pca", pca_dim=16)
    after = torch.rand(1)
    b = build(pair, device, type="pca", pca_dim=16)
    assert torch.equal(a.x, b.x) and torch.equal(a.y, b.y)
    assert torch.equal(after, expected_next)  # 呼び出しが RNG 状態を進めない


def test_pca_dim_is_clamped_with_warning(pair):
    f = build(pair, type="pca", pca_dim=64)  # raw 次元は 48
    assert f.dim == 48
    (w,) = f.warnings
    assert w.code == "PCA_DIM_REDUCED" and w.params == {"requested": 64, "actual": 48}


def test_position_channels_use_each_images_size():
    pa, ga = extract_patches(random_image(32, 64, 1), 16, 16)  # H=32, W=64 -> 2x4
    pb, gb = extract_patches(random_image(48, 32, 2), 16, 16)  # H=48, W=32 -> 3x2
    params = FeatureParams(type="color", normalize="none", position_weight=2.0)
    f = build_features(pa, ga, pb, gb, params)
    for feat, grid in ((f.x, ga), (f.y, gb)):
        expected = 2.0 * patch_centers(grid) / torch.tensor([grid.image_width, grid.image_height])
        assert torch.allclose(feat[:, 3:], expected)
    assert f.x[0, 3:].tolist() == pytest.approx([2.0 * 8 / 64, 2.0 * 8 / 32])


def test_position_weight_zero_adds_no_channels(pair):
    assert build(pair, type="color", position_weight=0.0).dim == 3


def test_position_is_added_after_normalization(pair):
    f = build(pair, type="color", normalize="l2", position_weight=1.0)
    assert torch.allclose(f.x[:, :3].norm(dim=1), torch.ones(256), atol=1e-5)


def test_position_weight_pulls_matches_toward_same_location():
    """位置チャンネルが効くと、同一画像でも最近傍が自パッチ (対角) に寄る。"""
    img = torch.full((3, 32, 32), 0.5)  # 一様画像: 色だけでは全パッチが同一
    pa, ga = extract_patches(img, 8, 8)
    f = build_features(
        pa, ga, pa, ga, FeatureParams(type="color", normalize="none", position_weight=1.0)
    )
    assert torch.equal(torch.cdist(f.x, f.y).argmin(dim=1), torch.arange(ga.n))


@pytest.mark.parametrize("kw", [{"pca_dim": 0}, {"position_weight": -0.1}])
def test_invalid_params(pair, kw):
    with pytest.raises(ValueError):
        build(pair, **kw)


def test_mismatched_patch_dims_rejected():
    pa, ga = extract_patches(random_image(32, 32, 1), 4, 4)
    pb, gb = extract_patches(random_image(32, 32, 2), 8, 8)
    with pytest.raises(ValueError):
        build_features(pa, ga, pb, gb, FeatureParams())


def test_end_to_end_is_deterministic():
    """画像 → パッチ → 特徴 が、同じ入力から毎回同一の X を返す。"""

    def run():
        pa, ga = extract_patches(random_image(64, 48, 7), 16, 16)
        pb, gb = extract_patches(random_image(64, 48, 8), 16, 16)
        return build_features(pa, ga, pb, gb, FeatureParams(pca_dim=8, position_weight=0.3))

    a, b = run(), run()
    assert a.x.shape == (12, 10) and torch.equal(a.x, b.x) and torch.equal(a.y, b.y)
