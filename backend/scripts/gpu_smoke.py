"""flash-sinkhorn の GPU スモークテスト (SPEC 3.4)。

1024x1024 のランダム点群で SamplesLoss(potentials=True) を実行し、
f, g の形状・有限性と所要時間 (初回 = Triton JIT 込み / 2回目以降) を出力する。
"""

import time

import torch
from flash_sinkhorn import SamplesLoss

N = M = 1024
D = 64
BLUR = 0.05


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1000


def main() -> None:
    assert torch.cuda.is_available(), "CUDA is not available"
    dev = torch.device("cuda:0")
    print(
        f"torch {torch.__version__} / cuda {torch.version.cuda} / {torch.cuda.get_device_name(0)}"
    )

    g = torch.Generator(device=dev).manual_seed(0)
    x = torch.randn(N, D, device=dev, generator=g, dtype=torch.float32).contiguous()
    y = torch.randn(M, D, device=dev, generator=g, dtype=torch.float32).contiguous()
    a = torch.full((N,), 1.0 / N, device=dev)
    b = torch.full((M,), 1.0 / M, device=dev)

    solver = SamplesLoss(loss="sinkhorn", blur=BLUR, scaling=0.5, debias=False, potentials=True)

    (f, gg), t_first = timed(lambda: solver(a, x, b, y))
    times = [timed(lambda: solver(a, x, b, y))[1] for _ in range(10)]

    print(f"f: {tuple(f.shape)} {f.dtype}  g: {tuple(gg.shape)} {gg.dtype}")
    print(f"finite: f={bool(torch.isfinite(f).all())} g={bool(torch.isfinite(gg).all())}")
    print(f"first call (JIT incl.): {t_first:.1f} ms")
    print(
        f"warm  median/min/max  : {sorted(times)[len(times) // 2]:.1f} / {min(times):.1f} / {max(times):.1f} ms"
    )
    print(f"peak GPU mem: {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
