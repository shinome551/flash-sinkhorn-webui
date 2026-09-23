"""ハードマッチング: c-transform の argmin による 1 対 1 寄りの対応 (SPEC 3.6)。

``j*(i) = argmin_j [C_ij - g_j]``。計画の行 ``P_ij ∝ b_j exp((g_j - C_ij) / ε)`` の argmax と
``b`` が一様なら数式上一致する (= ソフト計画の top-1)。違いは表示量で、変位は分布の重心ではなく
``j*`` の中心を指し、受け取り量は「何個の A パッチに選ばれたか」になる。
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from app.ot.plan import auto_chunk_rows


@dataclass(frozen=True)
class HardAssignment:
    idx: Tensor  # [N] int64 j*(i)
    cost: Tensor  # [N] C_{i j*(i)}
    displacement: Tensor  # [N, 2] center_B(j*) - center_A(i)
    col_mass: Tensor  # [M] (Σ_{i: j*(i)=j} a_i) / b_j (col_mass と同じ規約)


@torch.no_grad()
def c_transform_dense(
    x: Tensor, y: Tensor, g: Tensor, *, cost_scale: float = 1.0, chunk_rows: int | None = None
) -> tuple[Tensor, Tensor]:
    """``(min_j [C_ij - g_j], argmin_j)`` を行チャンクで求める (CPU / GPU 共通のリファレンス)。"""
    n, m = x.shape[0], y.shape[0]
    rows = chunk_rows if chunk_rows is not None else auto_chunk_rows(n, m)
    values, idx = [], []
    for start in range(0, n, rows):
        sl = slice(start, min(start + rows, n))
        cost = cost_scale * torch.cdist(x[sl], y, compute_mode="donot_use_mm_for_euclid_dist") ** 2
        v, j = (cost - g[None, :]).min(dim=1)
        values.append(v)
        idx.append(j)
    return torch.cat(values), torch.cat(idx)


@torch.no_grad()
def summarize_hard(
    idx: Tensor,
    x: Tensor,
    y: Tensor,
    a: Tensor,
    b: Tensor,
    centers_a: Tensor,
    centers_b: Tensor,
    *,
    cost_scale: float = 1.0,
) -> HardAssignment:
    """argmin ``idx`` から表示用の量を作る。コストは c-transform の値でなく ``C_ij*`` を返す。"""
    cost = cost_scale * ((x - y[idx]) ** 2).sum(dim=1)
    received = torch.zeros(y.shape[0], dtype=torch.float64, device=y.device)
    received.index_add_(0, idx, a.double())
    return HardAssignment(
        idx=idx,
        cost=cost,
        displacement=centers_b[idx] - centers_a,
        col_mass=(received / b.double()).to(b.dtype),
    )
