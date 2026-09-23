"""O(nd) 重心射影 (``apply_plan_mat_flashstyle``) と行チャンク要約の照合 (SPEC 3.5)。GPU が無い環境では skip。"""

import pytest
import torch

from app.ot import SolveParams
from app.ot.plan import summarize_plan
from app.ot.projection import flash_barycentric_projection
from app.services.features import FeatureParams
from tests.synth import problem, shifted_pair

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

DISP_ATOL_PX = 0.05  # 実測の最大差は N=1024 で 0.007 px、N=4096 で 0.026 px
ROW_MASS_RTOL = 2e-3  # 実測 5e-4 程度 (N=4096 で 8e-4)


@pytest.fixture(scope="module")
def flash():
    from app.ot.flash_backend import FlashBackend

    return FlashBackend()


def check(p, f, g, eps, params):
    args = (p["x"], p["y"], f, g, p["a"], p["b"], eps, p["centers_a"], p["centers_b"])
    disp, row_mass = flash_barycentric_projection(*args, cost_scale=params.cost_scale)
    ref = summarize_plan(*args, cost_scale=params.cost_scale, chunk_rows=37)
    assert disp.shape == ref.displacement.shape and row_mass.shape == ref.row_mass.shape
    assert (disp - ref.displacement).norm(dim=1).max().item() < DISP_ATOL_PX
    assert torch.allclose(row_mass, ref.row_mass, rtol=ROW_MASS_RTOL, atol=0)


@pytest.mark.parametrize(
    "params",
    [
        SolveParams(),
        SolveParams(blur=0.2),
        SolveParams(half_cost=True),
        SolveParams(reach_x=0.5, reach_y=0.5),
        SolveParams(reach_x=0.5),
    ],
    ids=["default", "blur0.2", "half_cost", "unbalanced", "semi_unbalanced"],
)
def test_matches_chunked_summary(flash, params):
    a_img, b_img = shifted_pair(128, 128, 16, 8)  # patch 8 -> N=M=256
    p = problem(a_img, b_img, device="cuda")
    f, g, eps, _ = flash.solve_potentials(p["x"], p["y"], p["a"], p["b"], params)
    check(p, f, g, eps, params)


def test_rectangular_and_uneven_sizes(flash):
    """N≠M、かつ 16 の倍数でない (Triton のマスク境界)。"""
    a_img, b_img = shifted_pair(120, 136, 16, 8)
    p = problem(a_img, b_img[:, :104, :], device="cuda")  # N=15x17=255, M=13x17=221
    assert p["x"].shape[0] != p["y"].shape[0]
    params = SolveParams()
    f, g, eps, _ = flash.solve_potentials(p["x"], p["y"], p["a"], p["b"], params)
    check(p, f, g, eps, params)


@pytest.mark.parametrize("dim", [1, 2])
def test_low_dim_features_are_padded(flash, dim):
    """``mat`` の列 (中心 2 + 行質量 1) が特徴次元に収まらないときはゼロ埋めで対応する。"""
    a_img, b_img = shifted_pair(64, 64, 8, 8)
    p = problem(a_img, b_img, device="cuda", feature=FeatureParams(pca_dim=dim))
    assert p["x"].shape[1] == dim
    params = SolveParams(blur=0.2)
    f, g, eps, _ = flash.solve_potentials(p["x"], p["y"], p["a"], p["b"], params)
    check(p, f, g, eps, params)


def test_rejects_bad_centers(flash):
    a_img, b_img = shifted_pair(64, 64, 8, 8)
    p = problem(a_img, b_img, device="cuda")
    n, m = p["x"].shape[0], p["y"].shape[0]
    with pytest.raises(ValueError, match="centers"):
        flash_barycentric_projection(
            p["x"], p["y"], torch.zeros(n, device="cuda"), torch.zeros(m, device="cuda"),
            p["a"], p["b"], 0.01, p["centers_a"], p["centers_b"][:, :1],
        )  # fmt: skip
