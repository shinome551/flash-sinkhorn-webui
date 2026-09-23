"""end-to-end のベンチ (SPEC 8 章: 512px / patch 16 (N=M=768) で 1 秒以内)。

起動中のサーバに対して、同梱サンプル ``mirror`` (512×384 → N=M=768) で ``POST /api/match`` を
繰り返し、応答までの時間 (HTTP 往復、JSON の生成・転送込み) と ``stats.elapsed_ms`` の中央値を出す::

    uv run uvicorn app.main:app --port 8000   # 別端末
    uv run python -m scripts.bench_api [--url http://localhost:8000] [--repeat 10] [--backend flash]
"""

import argparse
import statistics
import time

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--sample", default="mirror")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--backend", choices=["auto", "flash", "dense"], default=None)
    args = parser.parse_args()

    with httpx.Client(base_url=args.url, timeout=120) as client:
        health = client.get("/api/health").json()
        pair = client.post(f"/api/samples/{args.sample}").raise_for_status().json()
        body = {"image_a": pair["a"]["image_id"], "image_b": pair["b"]["image_id"]}
        if args.backend:
            body["ot"] = {"backend": args.backend}

        client.post("/api/match", json=body).raise_for_status()  # ウォームアップ
        rtts, stages = [], []
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            resp = client.post("/api/match", json=body).raise_for_status()
            rtts.append((time.perf_counter() - t0) * 1e3)
            stages.append(resp.json()["stats"]["elapsed_ms"])
        stats = resp.json()["stats"]

    print(f"device: {health['gpu_name'] or health['device']}, backend: {stats['backend']}")
    print(f"N={stats['n_patches_a']} M={stats['n_patches_b']} d={stats['feature_dim']}, "
          f"iterations={stats['iterations']}, median of {args.repeat}")  # fmt: skip
    for key in stages[0]:
        print(f"  {key:>10}: {statistics.median(s[key] for s in stages):8.1f} ms")
    print(f"  {'round trip':>10}: {statistics.median(rtts):8.1f} ms (HTTP 往復)")


if __name__ == "__main__":
    main()
