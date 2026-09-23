"""``N×M`` コスト行列を実体化する log-domain Sinkhorn (SPEC 3.7)。CPU / GPU どちらでも動く。

flash-sinkhorn の ``sinkhorn_flashstyle_symmetric`` と同じ手順 (GeomLoss 流の対称更新):

1. ε₀ で全更新 (α=1) → 2. 各 ε で ``(1-α) f + α T(g)``, α=½ を最終 ε の追加反復まで
→ 3. 最終 ε で全更新 (extrapolation)。``T`` は softmin (unbalanced では ``1/(1+ε/ρ)`` 倍)。
用途は CUDA が無い環境、CI での数値検証、flash 出力との照合。
"""

import torch
from torch import Tensor

from app.ot.base import SolveInfo, SolveParams, check_inputs
from app.ot.hard import c_transform_dense
from app.ot.schedule import build_eps_list


class DenseBackend:
    name = "dense"

    def solve_potentials(
        self, x: Tensor, y: Tensor, a: Tensor, b: Tensor, params: SolveParams
    ) -> tuple[Tensor, Tensor, float, SolveInfo]:
        params.validate()
        check_inputs(x, y, a, b)
        f, g, iterations, converged = _sinkhorn(x, y, a, b, params)
        info = SolveInfo("dense", str(x.device), iterations, converged)
        return f, g, params.eps, info

    def divergence(
        self, x: Tensor, y: Tensor, a: Tensor, b: Tensor, params: SolveParams
    ) -> float | None:
        """``OT(a,b) - ½OT(a,a) - ½OT(b,b)`` (双対値 ``<a,f> + <b,g>`` で評価)。balanced のみ。"""
        if not params.balanced:
            return None

        def value(u: Tensor, v: Tensor, p: Tensor, q: Tensor) -> Tensor:
            f, g, _, _ = self.solve_potentials(u, v, p, q, params)
            return (p.double() * f.double()).sum() + (q.double() * g.double()).sum()

        return float(value(x, y, a, b) - 0.5 * value(x, x, a, a) - 0.5 * value(y, y, b, b))

    def c_transform(self, x: Tensor, y: Tensor, g: Tensor, params: SolveParams) -> Tensor:
        return c_transform_dense(x, y, g, cost_scale=params.cost_scale)[1]


def _damping(eps: float, reach: float | None) -> float:
    return 1.0 if reach is None else 1.0 / (1.0 + eps / reach**2)


def _sinkhorn(
    x: Tensor, y: Tensor, a: Tensor, b: Tensor, p: SolveParams
) -> tuple[Tensor, Tensor, int, bool]:
    # mm 展開を使う既定の cdist は桁落ちするので、小さい ε では厳密モードを使う
    cost = p.cost_scale * torch.cdist(x, y, compute_mode="donot_use_mm_for_euclid_dist") ** 2
    log_a, log_b = a.log(), b.log()
    eps_list = build_eps_list(x, y, p)

    def update(f: Tensor, g: Tensor, eps: float, alpha: float) -> tuple[Tensor, Tensor]:
        f_new = -eps * torch.logsumexp((g + eps * log_b)[None, :] / eps - cost / eps, dim=1)
        g_new = -eps * torch.logsumexp((f + eps * log_a)[:, None] / eps - cost / eps, dim=0)
        f_new *= _damping(eps, p.reach_x)
        g_new *= _damping(eps, p.reach_y)
        return (1 - alpha) * f + alpha * f_new, (1 - alpha) * g + alpha * g_new

    f = torch.zeros_like(a)
    g = torch.zeros_like(b)
    f, g = update(f, g, eps_list[0], 1.0)
    iterations = 1

    tol = None if p.threshold is None else p.threshold * p.eps
    prev: tuple[Tensor, Tensor] | None = None
    converged = False
    for k, eps in enumerate(eps_list, start=1):
        f, g = update(f, g, eps, 0.5)
        iterations += 1
        if tol is not None and k % p.inner_iterations == 0:
            if prev is not None and max((f - prev[0]).abs().max(), (g - prev[1]).abs().max()) < tol:
                converged = True
                break
            prev = (f.clone(), g.clone())

    f, g = update(f, g, eps_list[-1], 1.0)
    return f, g, iterations + 1, converged
