"""輸送計画の行チャンク要約 (SPEC 3.5)。

``P_ij = a_i b_j exp((f_i + g_j - C_ij) / eps)`` を ``N×M`` で持たず、行チャンクごとに
``log P`` を作って統計だけを取り出す。コスト ``C_ij = cost_scale * ||x_i - y_j||²`` は
ソルバと同じ規約でなければならない (``SolveParams.cost_scale`` を渡す)。
"""

from dataclasses import dataclass

import torch
from torch import Tensor

CHUNK_BYTES = 64 * 2**20  # チャンク行数 * M * 4 bytes の上限


@dataclass(frozen=True)
class PlanSummary:
    """A 側パッチ ``i`` ごとの要約 (``N`` 行) と B 側の受け取り量 (``M`` 列)。

    ``top_weight`` は条件付き確率 ``p̃_ij = P_ij / row_mass_i`` の上位 ``k`` 件 (降順)、
    ``top_cost`` はその ``C_ij``。``top_valid`` が False の要素は ``min_weight`` 未満で打ち切られた
    もの (``top_idx`` などの値自体は残してある)。
    """

    row_mass: Tensor  # [N] Σ_j P_ij (balanced なら ≈ a_i)
    top_idx: Tensor  # [N, k] int64
    top_weight: Tensor  # [N, k]
    top_cost: Tensor  # [N, k]
    top_valid: Tensor  # [N, k] bool
    entropy: Tensor  # [N] 打ち切り前の全列で計算した行エントロピー (nat)
    confidence: Tensor  # [N] max_j p̃_ij
    displacement: Tensor  # [N, 2] Σ_j p̃_ij center_B(j) - center_A(i)
    col_mass: Tensor  # [M] (Σ_i P_ij) / b_j (1.0 が期待値)


def auto_chunk_rows(n_rows: int, n_cols: int, chunk_bytes: int = CHUNK_BYTES) -> int:
    """``chunk * M * 4 bytes <= chunk_bytes`` を満たす最大の行数 (最低 1、``N`` 以下)。"""
    return max(1, min(n_rows, chunk_bytes // (n_cols * 4)))


@torch.no_grad()
def summarize_plan(
    x: Tensor,
    y: Tensor,
    f: Tensor,
    g: Tensor,
    a: Tensor,
    b: Tensor,
    eps: float,
    centers_a: Tensor,
    centers_b: Tensor,
    *,
    cost_scale: float = 1.0,
    top_k: int = 3,
    min_weight: float = 1e-4,
    chunk_rows: int | None = None,
) -> PlanSummary:
    """``chunk_rows`` を省略すると ``auto_chunk_rows`` で決める (テスト用に上書きできる)。"""
    n, m = x.shape[0], y.shape[0]
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1 (got {top_k})")
    if centers_a.shape != (n, 2) or centers_b.shape != (m, 2):
        raise ValueError("centers_a / centers_b must be [N, 2] / [M, 2]")
    k = min(top_k, m)
    rows = chunk_rows if chunk_rows is not None else auto_chunk_rows(n, m)
    if rows < 1:
        raise ValueError(f"chunk_rows must be >= 1 (got {rows})")

    log_a, log_b = a.log(), b.log()
    col_acc = torch.zeros(m, dtype=torch.float64, device=y.device)
    parts: dict[str, list[Tensor]] = {
        key: [] for key in ("row_mass", "idx", "weight", "cost", "entropy", "disp")
    }

    for start in range(0, n, rows):
        sl = slice(start, min(start + rows, n))
        # mm 展開の cdist は桁落ちするので厳密モード (ソルバのコストと一致させる)
        cost = cost_scale * torch.cdist(x[sl], y, compute_mode="donot_use_mm_for_euclid_dist") ** 2
        log_p = (f[sl, None] + g[None, :] - cost) / eps + log_a[sl, None] + log_b[None, :]
        lse = torch.logsumexp(log_p, dim=1)  # log row_mass
        col_acc += log_p.exp().sum(dim=0, dtype=torch.float64)

        cond = (log_p - lse[:, None]).exp()  # p̃: 各行の和が 1
        weight, idx = torch.topk(cond, k, dim=1)
        parts["row_mass"].append(lse.exp())
        parts["idx"].append(idx)
        parts["weight"].append(weight)
        parts["cost"].append(cost.gather(1, idx))
        parts["entropy"].append(torch.special.entr(cond).sum(dim=1))
        parts["disp"].append(cond @ centers_b - centers_a[sl])

    weight = torch.cat(parts["weight"])
    return PlanSummary(
        row_mass=torch.cat(parts["row_mass"]),
        top_idx=torch.cat(parts["idx"]),
        top_weight=weight,
        top_cost=torch.cat(parts["cost"]),
        top_valid=weight >= min_weight,
        entropy=torch.cat(parts["entropy"]),
        confidence=weight[:, 0],
        displacement=torch.cat(parts["disp"]),
        col_mass=(col_acc / b.double()).to(b.dtype),
    )
