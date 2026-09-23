import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes import health, images, match, samples
from app.config import Settings
from app.errors import AppError
from app.schemas.common import ErrorCode, ErrorDetail, ErrorResponse
from app.services.device import resolve_backend
from app.services.image_store import ImageStore

logger = logging.getLogger(__name__)


def _error_response(
    status_code: int, code: ErrorCode, message: str, hint: str | None = None
) -> JSONResponse:
    body = ErrorResponse(detail=ErrorDetail(code=code, message=message, hint=hint))
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


async def _handle_app_error(_: Request, exc: AppError) -> JSONResponse:
    return _error_response(exc.status_code, exc.code, exc.message, exc.hint)


async def _handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    errors = exc.errors()
    summary = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in errors[:5])
    if len(errors) > 5:
        summary += f" (+{len(errors) - 5} more)"
    return _error_response(422, ErrorCode.INVALID_PARAMS, summary, "Check the request parameters.")


async def _handle_http_exception(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    return _error_response(exc.status_code, ErrorCode.HTTP_ERROR, str(exc.detail))


async def _handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception", exc_info=exc)
    return _error_response(500, ErrorCode.INTERNAL_ERROR, "Internal server error.")


async def _warmup_flash() -> None:
    """flash が使えるとき Triton カーネルを事前コンパイルする。失敗しても起動は続ける。"""
    if resolve_backend("auto") != "flash":
        return
    try:
        from app.ot import get_backend
        from app.ot.flash_backend import FlashBackend

        backend = get_backend("flash")
        if isinstance(backend, FlashBackend):
            await run_in_threadpool(backend.warmup)
    except Exception:
        logger.exception("flash warmup failed; the first request will compile the kernels")


def _configure_logging() -> None:
    """uvicorn はルートロガーを設定しないので、``app.*`` の INFO を出せるようにする。"""
    app_logger = logging.getLogger("app")
    app_logger.setLevel(logging.INFO)
    if not logging.getLogger().handlers and not app_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        app_logger.addHandler(handler)


def create_app(settings: Settings | None = None) -> FastAPI:
    _configure_logging()
    settings = settings or Settings()
    store = ImageStore(settings.image_dir, settings.image_ttl_seconds)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        store.sweep()
        if settings.warmup_on_startup:
            await _warmup_flash()
        yield

    app = FastAPI(title="flash-sinkhorn-webui", lifespan=lifespan)
    app.state.settings = settings
    app.state.image_store = store
    app.state.gpu_lock = asyncio.Semaphore(1)  # GPU 計算は同時 1 件 (SPEC 8章)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.add_exception_handler(AppError, _handle_app_error)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, _handle_validation_error)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, _handle_http_exception)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _handle_unexpected)

    app.include_router(health.router, prefix="/api")
    app.include_router(images.router, prefix="/api")
    app.include_router(match.router, prefix="/api")
    app.include_router(samples.router, prefix="/api")
    # ルーターより後に mount するので /api が優先される
    if settings.static_dir is not None:
        app.mount("/", StaticFiles(directory=settings.static_dir, html=True), name="static")
    return app


app = create_app()
