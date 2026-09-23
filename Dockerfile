# syntax=docker/dockerfile:1

# ---- frontend: 静的ファイルをビルド ----
FROM node:22-slim AS frontend
WORKDIR /src/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- backend: FastAPI + ビルド済みフロント ----
# torch の cu128 wheel は CUDA ランタイムを同梱するので、ベースは CUDA イメージでなくてよい
# (GPU ドライバはホストから NVIDIA Container Toolkit 経由で入る)
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /bin/uv

# Triton はカーネルのランチャーを実行時に gcc でビルドする
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    TRITON_CACHE_DIR=/cache/triton \
    FSW_STATIC_DIR=/app/frontend/dist \
    FSW_TMP_DIR=/tmp/flash-sinkhorn-webui

WORKDIR /app/backend
COPY backend/pyproject.toml backend/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra deep

COPY backend/app ./app
COPY assets /app/assets
COPY --from=frontend /src/frontend/dist /app/frontend/dist

EXPOSE 8000
# 起動時の Triton コンパイル (初回 30 秒程度) が終わるまで health は応答しない
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
