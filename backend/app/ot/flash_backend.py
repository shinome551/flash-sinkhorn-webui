"""flash-sinkhorn (Triton, CUDA 必須) によるバックエンド (SPEC 3.4)。

ポテンシャルは ``SamplesLoss(potentials=True)`` ではなく、その内部の
``sinkhorn_flashstyle_symmetric`` を直接呼んで得る。理由 (0.3.3.post1 で実測):

- ``SamplesLoss`` の potentials 経路は ε スケジュール長 (約 10〜20 反復) で打ち切られ、
  再構成した計画の行質量が ``a_i`` の 0.8〜5 倍になる。反復数を延ばす引数 (``eps_list``) は
  この経路では無視され、``threshold`` は絶対値なので ε が小さいと収束前に止まる。
- ``sinkhorn_flashstyle_symmetric`` は ``eps_list`` を受け取れるため、最終 ε での追加反復と
  ε 相対の ``threshold`` (``SolveParams`` 参照) を使える。

``SamplesLoss`` はダイバージェンス (``debias=True``) の取得にのみ使う。TF32 は無効
(有効だと小さい ε で行質量誤差が約 10% に頭打ちになる)、autotune も無効
(有効だと ``N``/``M`` を 32 で割った値ごとに Triton の再チューニングが走り、数秒〜数十秒かかる。
無効でも実行時間は同等)。
"""

import inspect
import logging
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from app.ot.base import BackendUnavailableError, SolveInfo, SolveParams, check_inputs
from app.ot.schedule import build_eps_list

logger = logging.getLogger(__name__)

WARMUP_DIM = 64  # 既定の特徴 (pca_dim=64)

# 内部関数に依存するので、必要な引数が無ければ黙って別の挙動になる前に検出する
_REQUIRED_SOLVER_ARGS = {
    "eps_list", "cost_scale", "reach_x", "reach_y", "threshold", "check_every",
    "allow_tf32", "autotune", "return_n_iters",
}  # fmt: skip


def _load_solver() -> Callable[..., Any]:
    try:
        from flash_sinkhorn.sinkhorn_solvers import sinkhorn_flashstyle_symmetric
    except ImportError as e:
        raise BackendUnavailableError(f"flash_sinkhorn is not installed: {e}") from e
    missing = _REQUIRED_SOLVER_ARGS - set(
        inspect.signature(sinkhorn_flashstyle_symmetric).parameters
    )
    if missing:
        raise BackendUnavailableError(
            f"flash_sinkhorn is incompatible: sinkhorn_flashstyle_symmetric lacks {sorted(missing)}"
        )
    return sinkhorn_flashstyle_symmetric  # type: ignore[no-any-return]


def _require_cuda(*tensors: Tensor) -> None:
    if not torch.cuda.is_available():
        raise BackendUnavailableError("The flash backend requires CUDA, but it is not available.")
    for t in tensors:
        if not t.is_cuda:
            raise BackendUnavailableError(
                f"The flash backend requires CUDA tensors (got device={t.device})."
            )


class FlashBackend:
    name = "flash"

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise BackendUnavailableError(
                "The flash backend requires CUDA, but it is not available."
            )
        self._solver = _load_solver()

    def solve_potentials(
        self, x: Tensor, y: Tensor, a: Tensor, b: Tensor, params: SolveParams
    ) -> tuple[Tensor, Tensor, float, SolveInfo]:
        params.validate()
        check_inputs(x, y, a, b)
        _require_cuda(x, y, a, b)
        eps_list = build_eps_list(x, y, params)
        tol = None if params.threshold is None else params.threshold * params.eps
        f, g, iterations = self._solver(
            x, y, a, b,
            eps_list=eps_list,
            cost_scale=params.cost_scale,
            reach_x=params.reach_x,
            reach_y=params.reach_y,
            threshold=tol,
            check_every=params.inner_iterations,
            allow_tf32=False,
            autotune=False,
            return_n_iters=True,
        )  # fmt: skip
        # 反復の総数は 初回更新 1 + スケジュール + 追加反復 + extrapolation 1。早く終われば収束
        converged = iterations < len(eps_list) + 2
        info = SolveInfo("flash", str(x.device), int(iterations), converged)
        return f, g, params.eps, info

    def divergence(
        self, x: Tensor, y: Tensor, a: Tensor, b: Tensor, params: SolveParams
    ) -> float | None:
        """``SamplesLoss(debias=True)`` の値。ε スケジュールの反復だけで打ち切られる近似値。"""
        params.validate()
        check_inputs(x, y, a, b)
        _require_cuda(x, y, a, b)
        from flash_sinkhorn import SamplesLoss

        loss = SamplesLoss(
            "sinkhorn",
            blur=params.blur,
            scaling=params.scaling,
            debias=True,
            potentials=False,
            half_cost=params.half_cost,
            reach_x=params.reach_x,
            reach_y=params.reach_y,
            allow_tf32=False,
            autotune=False,
        )
        return float(loss(a, x, b, y).item())

    def c_transform(self, x: Tensor, y: Tensor, g: Tensor, params: SolveParams) -> Tensor:
        """``flash_sinkhorn.c_transform_fwd`` の argmin (Triton。``N×M`` を実体化しない)。"""
        _require_cuda(x, y, g)
        from flash_sinkhorn import c_transform_fwd

        _, idx = c_transform_fwd(
            x, y, g, cost_scale=params.cost_scale, allow_tf32=False, autotune=False
        )
        return idx

    def warmup(self) -> None:
        """Triton カーネルを事前コンパイルする。初回呼び出しは数秒かかるため起動時に呼ぶ。

        Triton は特徴次元 ``d`` ごとにカーネルを作り、整数引数が 16 で割り切れるかでも特殊化する。
        既定の特徴 (pca 64 次元) について、``N``/``M`` の割り切れる/割り切れない全組み合わせを踏む。
        それ以外の ``d`` は初回だけ 1〜2 秒かかる。
        """
        params = SolveParams(max_final_iters=0)
        for n, m in ((256, 256), (250, 256), (256, 250), (250, 250)):
            x = torch.randn(n, WARMUP_DIM, device="cuda")
            y = torch.randn(m, WARMUP_DIM, device="cuda")
            a = torch.full((n,), 1.0 / n, device="cuda")
            b = torch.full((m,), 1.0 / m, device="cuda")
            _, g, _, _ = self.solve_potentials(x, y, a, b, params)
            self.divergence(x, y, a, b, params)
            self.c_transform(x, y, g, params)
        torch.cuda.synchronize()
        logger.info("flash backend warmed up")
