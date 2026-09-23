"""ε スケジューリング。flash-sinkhorn (= GeomLoss) と同一の系列を CPU でも作れるよう再実装する。"""

import math

import numpy as np
from torch import Tensor

from app.ot.base import SolveParams


def max_diameter(x: Tensor, y: Tensor) -> float:
    """``x``, ``y`` を合わせたバウンディングボックスの対角線長。"""
    lo = x.min(dim=0).values.minimum(y.min(dim=0).values)
    hi = x.max(dim=0).values.maximum(y.max(dim=0).values)
    return float((hi - lo).norm().item())


def epsilon_schedule(diameter: float, blur: float, scaling: float) -> list[float]:
    """``diameter²`` から ``blur²`` へ、反復ごとに ``scaling²`` 倍で減衰する ε の系列。"""
    eps_list = [diameter**2]
    eps_list += [
        float(np.exp(e))
        for e in np.arange(2 * math.log(diameter), 2 * math.log(blur), 2 * math.log(scaling))
    ]
    eps_list.append(blur**2)
    return eps_list


def build_eps_list(x: Tensor, y: Tensor, params: SolveParams) -> list[float]:
    """ε スケジュールに、最終 ε での追加反復 (上限) を連結した系列。"""
    # 全点が同一など退化した入力では diameter=0 になる。blur 以下に頭打ちして log(0) を避ける
    diameter = max(max_diameter(x, y), params.blur)
    schedule = epsilon_schedule(diameter, params.blur, params.scaling)
    return schedule + [params.eps] * params.max_final_iters
