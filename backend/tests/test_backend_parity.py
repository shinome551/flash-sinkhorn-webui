"""flash と dense の整合 (SPEC 9)。GPU が無い環境では skip。"""

import pytest
import torch

from app.ot import DenseBackend, SolveParams
from app.ot.plan import summarize_plan
from tests.synth import problem, shifted_pair

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture(scope="module")
def prob():
    a_img, b_img = shifted_pair(128, 128, 16, 8)  # patch 8 -> N=M=256
    return problem(a_img, b_img, device="cuda")


@pytest.fixture(scope="module")
def flash():
    from app.ot.flash_backend import FlashBackend

    return FlashBackend()


def solve(be, p, params):
    return be.solve_potentials(p["x"], p["y"], p["a"], p["b"], params)


@pytest.mark.parametrize(
    "params",
    [
        SolveParams(),
        SolveParams(blur=0.2),
        SolveParams(half_cost=True),
        SolveParams(reach_x=1.0, reach_y=1.0, max_final_iters=100),
    ],
    ids=["default", "blur0.2", "half_cost", "unbalanced"],
)
def test_potentials_and_plan_agree(flash, prob, params):
    ff, gf, eps_f, info_f = solve(flash, prob, params)
    fd, gd, eps_d, _ = solve(DenseBackend(), prob, params)
    assert eps_f == eps_d == pytest.approx(params.blur**2)
    assert info_f.backend == "flash" and info_f.device.startswith("cuda")
    if params.balanced:
        # ポテンシャルは定数の移動 (f+c, g-c) を除いて一意
        shift = (ff - fd).mean()
        assert (ff - fd - shift).abs().max() < 1e-3
        assert (gf - gd + shift).abs().max() < 1e-3
    else:
        assert (ff - fd).abs().max() < 1e-3 and (gf - gd).abs().max() < 1e-3

    kw = {"cost_scale": params.cost_scale, "top_k": 3}
    summaries = [
        summarize_plan(prob["x"], prob["y"], f, g, prob["a"], prob["b"], eps_f,
                       prob["centers_a"], prob["centers_b"], **kw)
        for f, g in ((ff, gf), (fd, gd))
    ]  # fmt: skip
    agree = (summaries[0].top_idx[:, 0] == summaries[1].top_idx[:, 0]).float().mean().item()
    assert agree >= 0.99
    assert torch.allclose(summaries[0].row_mass, summaries[1].row_mass, rtol=5e-3)


def test_divergence_agrees(flash, prob):
    params = SolveParams(blur=0.2)
    args = (prob["x"], prob["y"], prob["a"], prob["b"], params)
    d_flash, d_dense = flash.divergence(*args), DenseBackend().divergence(*args)
    assert d_flash is not None and d_dense is not None and d_dense > 0
    assert d_flash == pytest.approx(d_dense, rel=0.05)


def test_flash_divergence_supports_unbalanced(flash, prob):
    params = SolveParams(blur=0.2, reach_x=1.0, reach_y=1.0)
    d = flash.divergence(prob["x"], prob["y"], prob["a"], prob["b"], params)
    assert d is not None and d > 0


def test_flash_rejects_cpu_tensors(flash, prob):
    from app.ot import BackendUnavailableError

    with pytest.raises(BackendUnavailableError, match="CUDA"):
        flash.solve_potentials(
            prob["x"].cpu(), prob["y"].cpu(), prob["a"].cpu(), prob["b"].cpu(), SolveParams()
        )


def test_auto_selects_flash_on_cuda():
    from app.ot import get_backend

    assert get_backend("auto").name == "flash"
    assert get_backend("dense").name == "dense"


@pytest.mark.parametrize(
    "params", [SolveParams(), SolveParams(half_cost=True)], ids=["default", "half_cost"]
)
def test_c_transform_agrees(flash, prob, params):
    _, g, _, _ = solve(flash, prob, params)
    idx_flash = flash.c_transform(prob["x"], prob["y"], g, params)
    idx_dense = DenseBackend().c_transform(prob["x"], prob["y"], g, params)
    assert idx_flash.dtype == torch.int64 and idx_flash.shape == (prob["x"].shape[0],)
    assert (idx_flash == idx_dense).float().mean().item() >= 0.99


def test_flash_c_transform_rejects_cpu_tensors(flash, prob):
    from app.ot import BackendUnavailableError

    with pytest.raises(BackendUnavailableError, match="CUDA"):
        flash.c_transform(
            prob["x"].cpu(), prob["y"].cpu(), torch.zeros(prob["y"].shape[0]), SolveParams()
        )
