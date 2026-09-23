import math
from itertools import pairwise

import pytest
import torch

from app.ot import BackendUnavailableError, DenseBackend, SolveParams, get_backend
from app.ot.schedule import build_eps_list, epsilon_schedule, max_diameter
from tests.synth import problem, shifted_pair


@pytest.fixture(scope="module")
def prob():
    a_img, b_img = shifted_pair(64, 64, 8, 8)  # patch 8 -> N=M=64
    return problem(a_img, b_img)


def plan_marginals(p, f, g, eps, cost_scale=1.0):
    cost = cost_scale * torch.cdist(p["x"].double(), p["y"].double()) ** 2
    plan = ((f.double()[:, None] + g.double()[None] - cost) / eps).exp()
    plan = plan * p["a"].double()[:, None] * p["b"].double()[None]
    return plan.sum(1), plan.sum(0)


def test_balanced_marginals_converge(prob):
    # 収縮が遅いテクスチャ問題では、既定の反復上限 (500) で行質量誤差 1% 未満 (blur=0.2)
    f, g, eps, info = DenseBackend().solve_potentials(
        prob["x"], prob["y"], prob["a"], prob["b"], SolveParams(blur=0.2)
    )
    rows, cols = plan_marginals(prob, f, g, eps)
    assert torch.allclose(rows, prob["a"].double(), rtol=1e-2)
    assert torch.allclose(cols, prob["b"].double(), rtol=1e-2)
    assert info.backend == "dense" and info.device == "cpu"


@pytest.mark.parametrize("half_cost", [False, True])
def test_eps_is_blur_squared_regardless_of_cost_convention(prob, half_cost):
    p = SolveParams(blur=0.2, half_cost=half_cost)
    f, g, eps, _ = DenseBackend().solve_potentials(prob["x"], prob["y"], prob["a"], prob["b"], p)
    assert eps == pytest.approx(0.04)
    # 計画は solver と同じコスト規約で再構成したときだけ周辺分布を満たす
    rows, _ = plan_marginals(prob, f, g, eps, cost_scale=p.cost_scale)
    assert torch.allclose(rows, prob["a"].double(), rtol=1e-2)


def test_wrong_cost_convention_breaks_marginals(prob):
    p = SolveParams(blur=0.2, half_cost=True)
    f, g, eps, _ = DenseBackend().solve_potentials(prob["x"], prob["y"], prob["a"], prob["b"], p)
    rows, _ = plan_marginals(prob, f, g, eps, cost_scale=1.0)
    assert not torch.allclose(rows, prob["a"].double(), rtol=1e-2)


def test_unbalanced_relaxes_marginal_and_large_reach_recovers_balanced(prob):
    be, args = DenseBackend(), (prob["x"], prob["y"], prob["a"], prob["b"])
    f, g, eps, _ = be.solve_potentials(*args, SolveParams(blur=0.2, reach_x=0.3, reach_y=0.3))
    rows, _ = plan_marginals(prob, f, g, eps)
    assert not torch.allclose(rows, prob["a"].double(), rtol=1e-2)

    f, g, eps, _ = be.solve_potentials(*args, SolveParams(blur=0.2, reach_x=1e3, reach_y=1e3))
    rows, cols = plan_marginals(prob, f, g, eps)
    assert torch.allclose(rows, prob["a"].double(), rtol=2e-2)
    assert torch.allclose(cols, prob["b"].double(), rtol=2e-2)


def test_threshold_stops_early_and_none_runs_to_the_cap(prob):
    be, args = DenseBackend(), (prob["x"], prob["y"], prob["a"], prob["b"])
    loose = SolveParams(blur=0.2, max_final_iters=200, threshold=1.0)  # 打ち切り幅 = 1.0 * eps
    _, _, _, early = be.solve_potentials(*args, loose)
    assert early.converged and early.iterations < 100
    tight = SolveParams(blur=0.2, max_final_iters=200, threshold=1e-9)
    _, _, _, capped = be.solve_potentials(*args, tight)
    n_eps = len(build_eps_list(prob["x"], prob["y"], tight))
    assert not capped.converged and capped.iterations == n_eps + 2
    no_stop = SolveParams(blur=0.2, max_final_iters=200, threshold=None)
    _, _, _, full = be.solve_potentials(*args, no_stop)
    assert not full.converged and full.iterations == n_eps + 2


def test_divergence_is_zero_for_identical_clouds_and_positive_otherwise(prob):
    be, p = DenseBackend(), SolveParams(blur=0.2)
    same = be.divergence(prob["x"], prob["x"], prob["a"], prob["a"], p)
    diff = be.divergence(prob["x"], prob["y"], prob["a"], prob["b"], p)
    assert same is not None and diff is not None
    assert abs(same) < 1e-4
    assert diff > 1e-2


def test_divergence_unavailable_for_unbalanced(prob):
    p = SolveParams(reach_x=1.0)
    assert DenseBackend().divergence(prob["x"], prob["y"], prob["a"], prob["b"], p) is None


@pytest.mark.parametrize(
    "kw",
    [
        {"blur": 0.0}, {"scaling": 1.0}, {"scaling": 0.0}, {"reach_x": 0.0},
        {"threshold": 0.0}, {"inner_iterations": 0}, {"max_final_iters": -1},
    ],
)  # fmt: skip
def test_invalid_params_are_rejected(prob, kw):
    with pytest.raises(ValueError):
        DenseBackend().solve_potentials(
            prob["x"], prob["y"], prob["a"], prob["b"], SolveParams(**kw)
        )


def test_shape_mismatch_is_rejected(prob):
    with pytest.raises(ValueError):
        DenseBackend().solve_potentials(
            prob["x"], prob["y"], prob["a"][:-1], prob["b"], SolveParams()
        )


# --- ε スケジュール ---


def test_epsilon_schedule_goes_from_diameter_squared_to_blur_squared():
    sched = epsilon_schedule(4.0, 0.05, 0.5)
    assert sched[0] == pytest.approx(16.0) and sched[-1] == pytest.approx(0.0025)
    assert all(a > b for a, b in pairwise(sched))
    ratios = [b / a for a, b in pairwise(sched[1:-1])]
    assert ratios == pytest.approx([0.25] * len(ratios))  # 各段で scaling² 倍


def test_epsilon_schedule_matches_flash_sinkhorn():
    common = pytest.importorskip("flash_sinkhorn.kernels._common")
    for diameter, blur, scaling in [(6.88, 0.05, 0.5), (55.0, 0.05, 0.7), (1.0, 0.2, 0.3)]:
        assert epsilon_schedule(diameter, blur, scaling) == pytest.approx(
            list(common.epsilon_schedule(diameter, blur, scaling))
        )


def test_eps_list_appends_final_iterations_and_survives_degenerate_input():
    x = torch.zeros(4, 3)
    el = build_eps_list(x, x, SolveParams(blur=0.1, max_final_iters=5))
    assert el[-6:] == [pytest.approx(0.01)] * 6  # スケジュール末尾 + 追加 5 回
    assert all(math.isfinite(e) and e > 0 for e in el)  # diameter=0 でも log(0) にならない


def test_max_diameter_is_bounding_box_diagonal():
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    y = torch.tensor([[0.0, 2.0], [0.5, 0.5]])
    assert max_diameter(x, y) == pytest.approx(math.sqrt(1.0 + 4.0))


# --- バックエンド選択 ---


@pytest.fixture
def no_cuda(monkeypatch):
    import app.ot
    from app.services.device import get_device_info

    monkeypatch.setattr(app.ot, "_instances", {})  # 他のテストが作った flash をキャッシュから外す
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    get_device_info.cache_clear()
    yield
    get_device_info.cache_clear()


def test_auto_falls_back_to_dense_without_cuda(no_cuda):
    assert get_backend("auto").name == "dense"


def test_explicit_flash_without_cuda_raises(no_cuda):
    with pytest.raises(BackendUnavailableError, match="CUDA"):
        get_backend("flash")
