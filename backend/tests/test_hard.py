import pytest
import torch

from app.ot import DenseBackend, SolveParams
from app.ot.hard import c_transform_dense, summarize_hard
from app.ot.plan import summarize_plan
from tests.synth import problem, shifted_pair, true_targets


@pytest.fixture(scope="module")
def shifted():
    """B(p) = A(p + (16, 8))。patch 8 -> 12x16 = 192 パッチ。"""
    a_img, b_img = shifted_pair(96, 128, 16, 8)
    p = problem(a_img, b_img)
    p["truth"] = true_targets(p["grid_a"], p["grid_b"], 16, 8)
    return p


PARAMS = SolveParams()


def solve(p, params=PARAMS):
    return DenseBackend().solve_potentials(p["x"], p["y"], p["a"], p["b"], params)


def hard(p, g, params=PARAMS):
    idx = DenseBackend().c_transform(p["x"], p["y"], g, params)
    return summarize_hard(
        idx, p["x"], p["y"], p["a"], p["b"], p["centers_a"], p["centers_b"],
        cost_scale=params.cost_scale,
    )  # fmt: skip


def test_c_transform_matches_brute_force(shifted):
    _, g, _, _ = solve(shifted)
    for scale in (1.0, 0.5):
        values, idx = c_transform_dense(shifted["x"], shifted["y"], g, cost_scale=scale)
        full = scale * torch.cdist(shifted["x"], shifted["y"]) ** 2 - g[None, :]
        assert torch.allclose(values, full.min(dim=1).values, atol=1e-5)
        assert torch.equal(idx, full.argmin(dim=1))
        chunked = c_transform_dense(shifted["x"], shifted["y"], g, cost_scale=scale, chunk_rows=7)
        assert torch.equal(chunked[1], idx)


@pytest.mark.parametrize(
    "params",
    [SolveParams(), SolveParams(half_cost=True), SolveParams(reach_x=1.0, reach_y=1.0)],
    ids=["default", "half_cost", "unbalanced"],
)
def test_hard_equals_soft_top1_for_uniform_weights(shifted, params):
    """``b`` が一様なら argmin_j [C_ij - g_j] は計画の行の argmax と一致する (SPEC 3.6)。"""
    p = shifted
    f, g, eps, _ = solve(p, params)
    soft = summarize_plan(
        p["x"], p["y"], f, g, p["a"], p["b"], eps, p["centers_a"], p["centers_b"],
        cost_scale=params.cost_scale,
    )  # fmt: skip
    h = hard(p, g, params)
    assert (h.idx == soft.top_idx[:, 0]).float().mean().item() >= 0.99
    same = h.idx == soft.top_idx[:, 0]
    assert torch.allclose(h.cost[same], soft.top_cost[same, 0], rtol=1e-4, atol=1e-5)


def test_hard_displacement_recovers_translation(shifted):
    _, g, _, _ = solve(shifted)
    h = hard(shifted, g)
    valid = shifted["truth"] >= 0
    assert (h.idx[valid] == shifted["truth"][valid]).float().mean().item() >= 0.9
    # 格子上の変位なので、正解の対応ではちょうど (-16, -8)
    exact = (h.displacement[valid] == torch.tensor([-16.0, -8.0])).all(dim=1)
    assert exact.float().mean().item() >= 0.9


def test_hard_col_mass_counts_assignments():
    n, m = 4, 2
    x = torch.tensor([[0.0], [0.1], [0.2], [5.0]])
    y = torch.tensor([[0.0], [5.0]])
    a, b = torch.full((n,), 1 / n), torch.full((m,), 1 / m)
    centers_a = torch.zeros(n, 2)
    centers_b = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    h = summarize_hard(torch.tensor([0, 0, 0, 1]), x, y, a, b, centers_a, centers_b)
    assert torch.allclose(h.col_mass, torch.tensor([1.5, 0.5]))  # (3/4) / (1/2), (1/4) / (1/2)
    assert torch.allclose((h.col_mass * b).sum(), a.sum())
    assert torch.allclose(h.cost, torch.tensor([0.0, 0.01, 0.04, 0.0]), atol=1e-6)
    assert torch.equal(h.displacement[3], torch.tensor([3.0, 4.0]))
