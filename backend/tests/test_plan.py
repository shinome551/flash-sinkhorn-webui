import math

import pytest
import torch

from app.ot import DenseBackend, SolveParams
from app.ot.plan import CHUNK_BYTES, PlanSummary, auto_chunk_rows, summarize_plan
from tests.synth import problem, shifted_pair, texture, true_targets

PARAMS = SolveParams()


def solve_and_summarize(p, params=PARAMS, **kw):
    f, g, eps, _ = DenseBackend().solve_potentials(p["x"], p["y"], p["a"], p["b"], params)
    return summarize_plan(
        p["x"], p["y"], f, g, p["a"], p["b"], eps, p["centers_a"], p["centers_b"],
        cost_scale=params.cost_scale, **kw,
    )  # fmt: skip


@pytest.fixture(scope="module")
def shifted():
    """96x128 の画像 A と、(dx, dy) = (16, 8) だけずらした B。patch 8 -> 12x16 = 192 パッチ。"""
    a_img, b_img = shifted_pair(96, 128, 16, 8)
    p = problem(a_img, b_img)
    p["truth"] = true_targets(p["grid_a"], p["grid_b"], 16, 8)
    return p


@pytest.fixture(scope="module")
def identical():
    img = texture(96, 128, seed=3)
    return problem(img, img.clone())


def assert_summaries_equal(s1: PlanSummary, s2: PlanSummary):
    for name in PlanSummary.__dataclass_fields__:
        t1, t2 = getattr(s1, name), getattr(s2, name)
        if t1.dtype == torch.bool or not t1.is_floating_point():
            assert torch.equal(t1, t2), name
        else:
            assert torch.allclose(t1, t2, rtol=1e-5, atol=1e-6), name


def test_chunked_equals_unchunked(shifted):
    n = shifted["x"].shape[0]
    whole = solve_and_summarize(shifted, chunk_rows=n)
    for rows in (1, 7, n - 1):
        assert_summaries_equal(whole, solve_and_summarize(shifted, chunk_rows=rows))
    assert_summaries_equal(whole, solve_and_summarize(shifted))  # 自動決定


def test_auto_chunk_rows_respects_memory_budget():
    for n, m in [(768, 768), (4096, 4096), (100_000, 4096), (10, 10**8)]:
        rows = auto_chunk_rows(n, m)
        assert 1 <= rows <= n
        assert rows * m * 4 <= CHUNK_BYTES or rows == 1
    assert auto_chunk_rows(4096, 4096) == 4096 and auto_chunk_rows(100_000, 4096) == 4096
    assert auto_chunk_rows(768, 768) == 768  # 小さい問題は 1 チャンク


def test_row_and_column_mass_match_weights(shifted):
    s = solve_and_summarize(shifted, SolveParams(blur=0.2))
    assert torch.allclose(s.row_mass, shifted["a"], rtol=1e-2)
    assert torch.allclose(s.col_mass, torch.ones_like(s.col_mass), rtol=1e-2)
    total = (s.col_mass * shifted["b"]).sum()
    assert total.item() == pytest.approx(s.row_mass.sum().item(), rel=1e-5)


def test_identical_images_match_the_diagonal(identical):
    s = solve_and_summarize(identical)
    n = identical["x"].shape[0]
    top1 = (s.top_idx[:, 0] == torch.arange(n)).float().mean().item()
    assert top1 >= 0.95
    assert s.displacement.abs().median().item() < 1.0  # ピクセル


def test_translation_is_recovered(shifted):
    """B(p) = A(p + (16, 8)) なので、A のパッチは B 上で (-16, -8) 動く。"""
    s = solve_and_summarize(shifted)
    valid = shifted["truth"] >= 0
    top1 = s.top_idx[valid, 0]
    assert (top1 == shifted["truth"][valid]).float().mean().item() >= 0.9
    disp = s.displacement[valid]
    err = (disp - torch.tensor([-16.0, -8.0])).norm(dim=1)
    assert err.median().item() <= shifted["grid_a"].patch_size  # 中央値誤差 1 パッチ以内


def test_output_invariants(shifted):
    s = solve_and_summarize(shifted, top_k=4)
    n, m = shifted["x"].shape[0], shifted["y"].shape[0]
    assert s.top_idx.shape == (n, 4) and s.top_weight.shape == (n, 4) and s.top_cost.shape == (n, 4)
    assert torch.all(s.top_weight[:, :-1] >= s.top_weight[:, 1:])  # 降順
    assert torch.all((s.top_weight >= 0) & (s.top_weight <= 1 + 1e-6))
    assert torch.equal(s.confidence, s.top_weight[:, 0])
    assert torch.all((s.entropy >= 0) & (s.entropy <= math.log(m) + 1e-5))
    assert torch.all((s.top_idx >= 0) & (s.top_idx < m))
    cost = torch.cdist(shifted["x"], shifted["y"]) ** 2
    assert torch.allclose(s.top_cost, cost.gather(1, s.top_idx), rtol=1e-4, atol=1e-5)


def test_top_cost_follows_half_cost_convention(shifted):
    s = solve_and_summarize(shifted, SolveParams(half_cost=True))
    cost = 0.5 * torch.cdist(shifted["x"], shifted["y"]) ** 2
    assert torch.allclose(s.top_cost, cost.gather(1, s.top_idx), rtol=1e-4, atol=1e-5)


def test_min_weight_masks_small_targets(shifted):
    everything = solve_and_summarize(shifted, top_k=5, min_weight=0.0)
    assert everything.top_valid.all()
    strict = solve_and_summarize(shifted, top_k=5, min_weight=0.5)
    assert torch.equal(strict.top_valid, strict.top_weight >= 0.5)
    assert strict.top_valid.sum() < everything.top_valid.sum()
    assert not solve_and_summarize(shifted, min_weight=2.0).top_valid.any()  # 0 件もありうる
    # 打ち切りはエントロピー・信頼度に影響しない (打ち切り前の全列で計算)
    assert torch.equal(strict.entropy, everything.entropy)


def test_uniform_plan_has_maximal_entropy_and_flat_confidence():
    n, m = 6, 10
    x, y = torch.zeros(n, 2), torch.zeros(m, 2)
    a, b = torch.full((n,), 1 / n), torch.full((m,), 1 / m)
    s = summarize_plan(
        x, y, torch.zeros(n), torch.zeros(m), a, b, 1.0,
        torch.zeros(n, 2), torch.arange(m * 2, dtype=torch.float32).view(m, 2),
    )  # fmt: skip
    assert torch.allclose(s.row_mass, a)
    assert torch.allclose(s.entropy, torch.full((n,), math.log(m)))
    assert torch.allclose(s.confidence, torch.full((n,), 1 / m))
    assert torch.allclose(s.displacement, torch.arange(m * 2.0).view(m, 2).mean(0).expand(n, 2))
    assert torch.allclose(s.col_mass, torch.ones(m))


def test_top_k_is_capped_by_number_of_targets():
    n, m = 3, 2
    x, y = torch.zeros(n, 2), torch.zeros(m, 2)
    a, b = torch.full((n,), 1 / n), torch.full((m,), 1 / m)
    s = summarize_plan(
        x,
        y,
        torch.zeros(n),
        torch.zeros(m),
        a,
        b,
        1.0,
        torch.zeros(n, 2),
        torch.zeros(m, 2),
        top_k=5,
    )
    assert s.top_idx.shape == (n, m)


def test_invalid_arguments_are_rejected():
    x = torch.zeros(3, 2)
    a = torch.full((3,), 1 / 3)
    args = (x, x, torch.zeros(3), torch.zeros(3), a, a, 1.0, torch.zeros(3, 2), torch.zeros(3, 2))
    with pytest.raises(ValueError):
        summarize_plan(*args, top_k=0)
    with pytest.raises(ValueError):
        summarize_plan(*args, chunk_rows=0)
    with pytest.raises(ValueError):
        summarize_plan(*args[:-1], torch.zeros(4, 2))
