"""OT コアのベンチ (Phase 3)。N=M=768 (512x384 / patch16) と 3072 (patch8) で solve / plan の所要時間を測る。

    uv run python -m scripts.bench_ot [--repeat 5] [--blur 0.05 0.2] [--backends flash dense] [--max-iters 500]

特徴は合成テクスチャ画像 (tests/synth.py) の 2 枚から通常の経路 (pca 64 次元 + zscore) で作る。
各設定で 1 回ウォームアップしてから ``--repeat`` 回の中央値を出す。行質量誤差は
``max_i |row_mass_i / a_i - 1|`` (収束の目安)。flash では O(nd) 重心射影 (``app.ot.projection``) の
所要時間と、チャンク要約との変位の最大差 (px) も出す (Phase 9)。
"""

import argparse
import statistics
import time

import torch

from app.ot import SolveParams, get_backend
from app.ot.plan import summarize_plan
from app.ot.projection import flash_barycentric_projection
from app.services.features import FeatureParams
from tests.synth import problem, shifted_pair


def timed(fn, device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1e3


def bench(
    name: str, patch: int, blur: float, backend_name: str, repeat: int, max_iters: int
) -> None:
    device = torch.device("cuda" if backend_name == "flash" or torch.cuda.is_available() else "cpu")
    img_a, img_b = shifted_pair(384, 512, 32, 16)
    p = problem(img_a, img_b, patch=patch, device=str(device), feature=FeatureParams(pca_dim=64))
    backend = get_backend(backend_name)  # type: ignore[arg-type]
    params = SolveParams(blur=blur, max_final_iters=max_iters)
    n = p["x"].shape[0]

    def solve():
        return backend.solve_potentials(p["x"], p["y"], p["a"], p["b"], params)

    def plan(f, g, eps):
        return summarize_plan(
            p["x"], p["y"], f, g, p["a"], p["b"], eps, p["centers_a"], p["centers_b"],
            cost_scale=params.cost_scale,
        )  # fmt: skip

    (f, g, eps, info), _ = timed(solve, device)  # warmup
    plan(f, g, eps)
    solve_ms = statistics.median(timed(solve, device)[1] for _ in range(repeat))
    plan_ms = statistics.median(timed(lambda: plan(f, g, eps), device)[1] for _ in range(repeat))
    div, div_ms = timed(lambda: backend.divergence(p["x"], p["y"], p["a"], p["b"], params), device)
    s = plan(f, g, eps)
    row_err = (s.row_mass / p["a"] - 1).abs().max().item()
    proj = ""
    if device.type == "cuda":

        def project():
            return flash_barycentric_projection(
                p["x"], p["y"], f, g, p["a"], p["b"], eps, p["centers_a"], p["centers_b"],
                cost_scale=params.cost_scale,
            )  # fmt: skip

        disp, _ = project()  # warmup
        proj_ms = statistics.median(timed(project, device)[1] for _ in range(repeat))
        diff = (disp - s.displacement).norm(dim=1).max().item()
        proj = f"  projection {proj_ms:6.1f} ms (diff {diff:.1e} px)"
    print(
        f"{name:10s} N=M={n:<5d} blur={blur:<5g} {backend_name:5s} "
        f"solve {solve_ms:8.1f} ms ({info.iterations} it, conv={info.converged})  "
        f"plan {plan_ms:7.1f} ms  divergence {div_ms:7.1f} ms ({div})  row_err {row_err:.1e}{proj}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--blur", type=float, nargs="+", default=[0.05, 0.2])
    ap.add_argument("--max-iters", type=int, default=SolveParams.max_final_iters)
    ap.add_argument("--backends", nargs="+", default=["flash", "dense"])
    ap.add_argument(
        "--sizes", type=int, nargs="+", default=[16, 8], help="patch sizes (768 / 3072)"
    )
    args = ap.parse_args()
    if torch.cuda.is_available():
        print(f"torch {torch.__version__} / {torch.cuda.get_device_name(0)}")
    for backend in args.backends:
        for patch in args.sizes:
            for blur in args.blur:
                bench(f"patch{patch}", patch, blur, backend, args.repeat, args.max_iters)


if __name__ == "__main__":
    main()
