"""``apply_plan_mat_flashstyle`` による O(nd) の重心射影 (SPEC 3.5 の代替経路)。

``summarize_plan`` (行チャンク) の変位と行質量を、``N×M`` を作らない Triton カーネルで独立に計算する。
top-k やエントロピーは行全体が要るのでこの経路では出せず、API では使わない (照合用のリファレンス)。

カーネルは ``mat`` に特徴次元と同じ列数 ``[M, d]`` を要求するので、列を
``[center_B(j) の x, y, 1, 0, ...]`` にして ``Σ_j P_ij center_B(j)`` と ``row_mass_i = Σ_j P_ij`` を
1 回で得る。``d < 3`` のときは ``x``, ``y`` をゼロ埋めする (コストは変わらない)。
"""

import torch
from torch import Tensor
from torch.nn import functional as F

_MIN_DIM = 3  # 中心座標 2 列 + 行質量 1 列


@torch.no_grad()
def flash_barycentric_projection(
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
) -> tuple[Tensor, Tensor]:
    """``(displacement[N, 2], row_mass[N])``。規約は ``summarize_plan`` と同じ (CUDA 必須)。"""
    from flash_sinkhorn.kernels.apply_flash import apply_plan_mat_flashstyle

    n, d = x.shape
    m = y.shape[0]
    if centers_a.shape != (n, 2) or centers_b.shape != (m, 2):
        raise ValueError("centers_a / centers_b must be [N, 2] / [M, 2]")
    if d < _MIN_DIM:
        x, y = F.pad(x, (0, _MIN_DIM - d)), F.pad(y, (0, _MIN_DIM - d))
        d = _MIN_DIM

    mat = torch.zeros(m, d, dtype=torch.float32, device=y.device)
    mat[:, :2] = centers_b
    mat[:, 2] = 1.0
    # カーネルは P_ij = a_i b_j exp((f̂_i + ĝ_j + 2 cost_scale x_i·y_j) / eps) の shifted 形式を取る
    f_hat = f - cost_scale * (x**2).sum(dim=1)
    g_hat = g - cost_scale * (y**2).sum(dim=1)
    out = apply_plan_mat_flashstyle(
        x.contiguous(), y.contiguous(), f_hat, g_hat, a.log(), b.log(), mat,
        eps=eps, axis=1, cost_scale=cost_scale, allow_tf32=False, autotune=False,
    )  # fmt: skip
    row_mass = out[:, 2]
    return out[:, :2] / row_mass[:, None] - centers_a, row_mass
