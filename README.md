# flash-sinkhorn-webui

English | [日本語](README.ja.md)

A web app that splits two images into patches, finds patch-to-patch correspondences with entropy-regularized optimal transport (Sinkhorn OT), and visualizes them interactively in the browser.
The OT solver is [flash-sinkhorn](https://pypi.org/project/flash-sinkhorn/) (PyTorch + Triton), which never materializes the N×M cost matrix.
Design details are in [docs/SPEC.md](docs/SPEC.md) and the development log is in [docs/DEVLOG.md](docs/DEVLOG.md) (both in Japanese). The UI is in Japanese; UI labels below are quoted with their meaning.

![A patch in image A is pinned, and lines connect it to its matches in image B](docs/screenshot.png)

- Backend: FastAPI (Python 3.12), PyTorch, flash-sinkhorn
- Frontend: React + Vite + TypeScript, Tailwind CSS, TanStack Query
- Runs on a CUDA GPU; falls back to a dense CPU solver when no GPU is available

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- Node.js 20.19+ / 22.12+ (required by Vite 8; tested with 24) and npm
- A CUDA GPU (optional). Without a GPU the app uses the dense CPU backend (up to 1024 patches, a few seconds per run)

## Run with Docker

```bash
docker compose up --build                                          # GPU (requires NVIDIA Container Toolkit)
docker compose -f compose.yaml -f compose.cpu.yaml up --build      # no GPU (dense CPU backend)
```

Open <http://localhost:8000> (the frontend is built and served by FastAPI). Change the port with `FSW_PORT=8080 docker compose up`.
The image includes every feature, DINOv2 included. It is several GB, and the first build takes a while to download torch.
DINOv2 weights and Triton compilation caches are kept in the named volume `cache`. Add other `FSW_*` settings to `environment` in `compose.yaml`.

If the GPU variant fails with `could not select device driver "nvidia"` or `failed to discover GPU vendor`, the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is missing on the host
(also required on WSL2 when using Docker Engine).

The rest of this section describes local development without containers.

## Setup

```bash
# backend (torch is the CUDA 12.8 build from download.pytorch.org; several GB)
cd backend
uv sync
# For deep features (feature.type = dinov2), run this instead. It adds timm and torchvision (cu128).
# A plain `uv sync` later removes the extra, so always pass `--extra deep`.
# uv sync --extra deep

# frontend
cd ../frontend
npm install
```

## Start

Use two terminals.

```bash
# 1. backend (http://localhost:8000)
cd backend
uv run uvicorn app.main:app --port 8000

# 2. frontend (http://localhost:5173)
cd frontend
npm run dev
```

Open <http://localhost:5173>. The badge at the top right turns green (GPU name) or yellow (no GPU) once the backend is reachable.
With a GPU, the backend compiles Triton kernels at startup (about 30 s the first time, a few seconds afterwards; `/api/health` does not respond until this finishes).

## Usage

1. Click a sample button next to 「サンプルで試す」 (Try a sample: shift / mirror / different photo), or drop images into slots A and B (PNG / JPEG / WebP, up to 10 MB; downscaled to 512 px on the long side).
2. Click 「実行」 (Run) in the header. If the run cannot start (no image, too many patches, etc.), the reason is shown below the button.
3. Hover over image A (or move with the arrow keys). The top-k matches in B are highlighted with opacity and connecting lines according to their weights.

Visualization modes:

| Mode | Shows |
|---|---|
| hover | Matches in B for the patch under the cursor in A |
| confidence / entropy | Colors A by each patch's confidence (maximum conditional probability) or row entropy. Low confidence = ambiguous match |
| flow | Arrows from each patch in A to the barycenter of its matches. 「確信度で薄く」 (fade by confidence) dims ambiguous arrows |
| coverage | Colors A by outgoing mass and B by received mass (1.0 = uniform on both sides). Shows the effect of unbalanced OT (reach) |
| pin | Click / Enter pins multiple patches for color-coded comparison (also works in confidence, entropy and coverage) |

The 「ソフト / ハード」 (soft / hard) toggle next to the modes switches the assignment type without rerunning. Hard assignment is the argmin of the c-transform, a single match per patch;
with uniform weights it points to the same patch as the soft top-1. The difference is in what is displayed: flow arrows point to the matched patch center instead of the barycenter,
and coverage becomes "how many A patches chose this B patch" (revealing many-to-one concentration and unchosen B patches).

Results can be saved as 「結果 JSON」 (request and response) and 「可視化 PNG」 (visualization image).

### Parameters

Each field has a detailed explanation behind the `?` in the UI. The main ones:

| Parameter | Default | Meaning |
|---|---|---|
| `patch.size` / `stride` | 16 / (size) | Patch side length and spacing [px]. Smaller is finer but increases the patch count N (512×384 at 16 px gives N = 768) |
| `feature.type` | `pca` | Patch vectorization. `raw` (pixels as-is) / `pca` (compressed to `pca_dim` dims with a basis shared by A and B) / `color` (mean RGB) / `dinov2` (384-dim DINOv2 ViT-S/14 tokens; needs `--extra deep`, see below) |
| `feature.normalize` | `zscore` | Standardize each dimension, then scale by 1/√d, so `blur` can be chosen independently of the feature dimension |
| `feature.position_weight` | 0 | Weight of the patch-center coordinates appended to the features. Larger values favor matches between nearby positions |
| `ot.blur` | 0.05 | Regularization strength (ε = blur²). Smaller gives sharper, closer to one-to-one matches but converges more slowly. 0.05–0.3 is a good range |
| `ot.scaling` | 0.5 | Shrink factor of ε-scaling. Closer to 1 is more stable but needs more iterations |
| `ot.reach_x` / `reach_y` | (balanced) | Setting a value relaxes mass conservation on the A / B side (unbalanced OT), allowing patches without a match (see below) |
| `ot.backend` | (server setting) | `auto` = flash if CUDA is available, otherwise dense. `flash` requires CUDA |
| `output.top_k` / `min_weight` | 3 / 1e-4 | Number of matches returned per patch, and the lower bound on conditional probability |

With `reach`, transport whose cost (squared feature distance) exceeds roughly reach² may be dropped along with its mass.
Measured on the samples (default parameters, zscore):

| reach (both x and y) | Shift | Different photo | Mirror (patch 24) |
|---|---|---|---|
| 1.0 | 97% | 64% | 66% |
| 0.5 | 91% | 33% | 42% |
| 0.2 | 74% | 4% | 23% |
| 0.1 | 44% | 0.2% | 12% |

(Total transported mass; balanced = 100%.) 0.2–1 is a good range; at 0.1 or below most mass is not transported even for well-matched pairs.
Check the result in coverage mode (A = outgoing mass, B = received mass) and the outgoing mass shown on hover. Flow arrows and pin lines fade for patches with little outgoing mass.
With only `reach_x` (semi-unbalanced), easily matched A patches may send several times 1/N to satisfy B's marginal.
The unbalanced divergence is not comparable to the balanced one because the transported mass differs (the dense backend does not compute it).

The stats panel warns when the row mass error is large (over 5% by default). This means the solver did not fully converge within the iteration limit.
The display is still valid because each row is normalized, but increasing `blur` helps.

`dinov2` uses intermediate representations of a self-supervised ViT, so it matches by *what* is in the image rather than how it looks.
The region covered by the grid is resized so that one stride equals one token (14 px), so changing the patch size also changes the resolution the model sees.
The first run downloads the weights (about 90 MB) from the Hugging Face Hub; after each server start, the first request takes 2–3 s to load the model.
Measured on the samples (patch 16, RTX 3060, fraction of hard assignments matching the ground-truth correspondence):

| Sample | pca | dinov2 |
|---|---|---|
| Shift | 98% | 78% |
| Mirror | 10% | 84% |

Feature extraction takes 20–30 ms after the first run (pca: about 10 ms). For pixel-level matches like a shift, pca is more accurate.

### Backend settings

Configure with environment variables (prefix `FSW_`) or `backend/.env`. The main ones:
`FSW_TMP_DIR` (image storage, defaults to the OS temp directory), `FSW_IMAGE_TTL_SECONDS` (3600),
`FSW_MAX_UPLOAD_BYTES` (10 MB), `FSW_RESIZE_MAX` (512), `FSW_MAX_PATCHES` (4096) / `FSW_MAX_PATCHES_CPU` (1024),
`FSW_DEFAULT_BACKEND` (`auto` / `flash` / `dense`), `FSW_WARN_ROW_MASS_ERROR` (0.05), `FSW_WARMUP_ON_STARTUP` (true).
See `backend/app/config.py` for the full list.

### API (curl)

```bash
curl localhost:8000/api/health
curl -F file=@a.png localhost:8000/api/images         # -> {"image_id": "...", "url": "/api/images/...", ...}
curl -X POST localhost:8000/api/samples/shift          # register a bundled sample -> {"a": {...}, "b": {...}}
curl -H 'Content-Type: application/json' localhost:8000/api/match \
     -d '{"image_a": "<id>", "image_b": "<id>", "ot": {"blur": 0.1}}'
```

See [SPEC section 4](docs/SPEC.md#4-api-仕様) or <http://localhost:8000/docs> (OpenAPI).

## Performance

Target from [SPEC section 8](docs/SPEC.md#8-非機能要件): end-to-end within 1 s for 512 px / patch 16 (N = M = 768) on a GPU.
Median of 10 runs on the bundled `mirror` sample (512×384) with default parameters (`uv run python -m scripts.bench_api`).

| Environment | Backend | Solve | Total (`stats.elapsed_ms.total`) | HTTP round trip |
|---|---|---|---|---|
| RTX 3060 12GB / torch 2.11.0+cu128 / flash-sinkhorn 0.3.3.post1 (WSL2) | flash | 120 ms | 152 ms | 176 ms |
| Same | dense (GPU) | 351 ms | 841 ms | 860 ms |
| Ryzen 5 5600 (6 cores), no GPU (`CUDA_VISIBLE_DEVICES=""`) | dense (CPU) | 2011 ms | 4179 ms | 4200 ms |

- flash meets the target (about 0.18 s). The solver runs up to the iteration limit (512).
- The dense total includes 461 ms (2188 ms on CPU) for the divergence, which solves three problems for debiasing. Disabling `compute_divergence` roughly halves it.
- The first run after startup takes a few hundred ms extra for initialization such as PCA.

## Development

```bash
cd backend  && uv run pytest               # tests (GPU-dependent tests are skipped without a GPU)
cd backend  && uv run ruff check . && uv run ruff format --check . && uv run mypy app tests scripts
cd frontend && npm test                    # vitest (unit tests for pure functions)
cd frontend && npm run lint                # oxlint
cd frontend && npm run format:check        # prettier (format with npm run format)
cd frontend && npm run build               # type check + build
cd frontend && npx playwright install chromium && npm run e2e   # E2E (starts backend / frontend automatically)
```

E2E reuses servers already running on ports 8000 / 5173.

Other scripts (run in `backend/`):

- `uv run python scripts/gpu_smoke.py`: checks that flash-sinkhorn runs on the GPU (1024 points; prints the shapes of `f, g` and the time).
- `uv run python -m scripts.bench_ot`: benchmark of the OT core alone. On flash it also reports the time of the O(nd) barycentric projection (SPEC 3.5) and its difference from the chunked summary.
- `uv run --with scikit-image python -m scripts.make_samples`: regenerates `assets/samples/` (image sources are listed in `assets/samples/CREDITS.md`).

## License

[MIT](LICENSE). See [assets/samples/CREDITS.md](assets/samples/CREDITS.md) for the sources and licenses of the bundled sample images.
