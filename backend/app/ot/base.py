"""OT ソルバの共通インタフェース (SPEC 3.4 / 3.7)。

コスト規約: ``C_ij = cost_scale * ||x_i - y_j||²`` (``half_cost`` なら ``cost_scale = 0.5``)。
最終 ε は ``blur²`` で、``half_cost`` でも変わらない。計画 ``P_ij = a_i b_j exp((f_i + g_j - C_ij) / ε)``
の再構成には、ここで返る ``(f, g, eps)`` と同じ規約のコストを使うこと。
"""

from dataclasses import dataclass
from typing import Protocol

from torch import Tensor


@dataclass(frozen=True)
class SolveParams:
    """Sinkhorn の数値パラメータ (バックエンド共通)。

    ``threshold`` は「``inner_iterations`` 反復ごとのポテンシャル最大変化量が
    ``threshold * ε`` 未満なら打ち切る」という **ε 相対** の値 (``None`` で打ち切りなし)。
    絶対値だと ε が小さいとき (blur=0.05 で ε=0.0025) 収束前に止まってしまうため。
    反復は ε スケジュール (``diameter²`` → ``blur²``) のあと、最終 ε で最大
    ``max_final_iters`` 回まで続く。
    """

    blur: float = 0.05
    scaling: float = 0.5
    half_cost: bool = False
    reach_x: float | None = None
    reach_y: float | None = None
    threshold: float | None = 1e-3
    inner_iterations: int = 10
    max_final_iters: int = 500

    @property
    def cost_scale(self) -> float:
        return 0.5 if self.half_cost else 1.0

    @property
    def eps(self) -> float:
        """ε スケジューリングの最終値。"""
        return self.blur**2

    @property
    def balanced(self) -> bool:
        return self.reach_x is None and self.reach_y is None

    def validate(self) -> None:
        if not self.blur > 0:
            raise ValueError(f"blur must be > 0 (got {self.blur})")
        if not 0 < self.scaling < 1:
            raise ValueError(f"scaling must be in (0, 1) (got {self.scaling})")
        for name, reach in (("reach_x", self.reach_x), ("reach_y", self.reach_y)):
            if reach is not None and not reach > 0:
                raise ValueError(f"{name} must be > 0 (got {reach})")
        if self.threshold is not None and not self.threshold > 0:
            raise ValueError(f"threshold must be > 0 or None (got {self.threshold})")
        if self.inner_iterations < 1:
            raise ValueError(f"inner_iterations must be >= 1 (got {self.inner_iterations})")
        if self.max_final_iters < 0:
            raise ValueError(f"max_final_iters must be >= 0 (got {self.max_final_iters})")


@dataclass(frozen=True)
class SolveInfo:
    backend: str
    device: str
    iterations: int
    converged: bool  # ``threshold`` により打ち切られたか (False なら反復上限まで回した)


class SinkhornBackend(Protocol):
    name: str

    def solve_potentials(
        self, x: Tensor, y: Tensor, a: Tensor, b: Tensor, params: SolveParams
    ) -> tuple[Tensor, Tensor, float, SolveInfo]:
        """``x[N,d]``, ``y[M,d]``, 重み ``a[N]``, ``b[M]`` から ``(f[N], g[M], eps, info)`` を返す。

        入力は同一デバイス上の float32 (contiguous)。``f``, ``g`` は debias なしの
        ポテンシャルで、輸送計画の再構成に使える。``eps`` は最終 ε (= ``blur²``)。
        """
        ...

    def divergence(
        self, x: Tensor, y: Tensor, a: Tensor, b: Tensor, params: SolveParams
    ) -> float | None:
        """Sinkhorn ダイバージェンス。バックエンドが未対応の設定 (dense の unbalanced) は ``None``。"""
        ...

    def c_transform(self, x: Tensor, y: Tensor, g: Tensor, params: SolveParams) -> Tensor:
        """ハードマッチング ``j*(i) = argmin_j [C_ij - g_j]`` (``[N]`` int64, SPEC 3.6)。"""
        ...


class BackendUnavailableError(RuntimeError):
    """要求されたバックエンドがこの環境で使えない (CUDA 無し、flash_sinkhorn 未導入など)。"""


def check_inputs(x: Tensor, y: Tensor, a: Tensor, b: Tensor) -> None:
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError(
            f"x and y must be [N, d] / [M, d] (got {tuple(x.shape)}, {tuple(y.shape)})"
        )
    if a.shape != (x.shape[0],) or b.shape != (y.shape[0],):
        raise ValueError("a and b must match the number of points in x and y")
    if not (x.device == y.device == a.device == b.device):
        raise ValueError("x, y, a, b must be on the same device")
    if x.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("x and y must not be empty")
