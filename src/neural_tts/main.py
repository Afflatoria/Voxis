"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from neural_tts.api.http import router as http_router
from neural_tts.api.websocket import router as ws_router
from neural_tts.config import get_settings
from neural_tts.logging_config import configure_logging, get_logger
from neural_tts.models.factory import create_backend
from neural_tts.streaming.scheduler import InferenceScheduler

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    backend = create_backend(settings)
    scheduler = InferenceScheduler(max_concurrent=settings.max_concurrent_sessions)

    app.state.backend = backend
    app.state.scheduler = scheduler

    logger.info("loading_backend", backend=settings.backend)
    try:
        await backend.load()
        info = backend.info()
        logger.info(
            "backend_ready",
            name=info.name,
            device=info.device,
            sample_rate=info.sample_rate,
            load_time_seconds=info.load_time_seconds,
        )
    except Exception as exc:
        logger.error("backend_load_failed", error=str(exc))
        # Keep server up so health endpoint reports not-ready (useful for setup/debug).

    yield

    await backend.aclose()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Neural TTS",
        version="0.1.0",
        description="Local real-time streaming TTS (M1 baseline)",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(http_router)
    app.include_router(ws_router)
    return app


app = create_app()


def cli() -> None:
    """Entry point for `neural-tts` console script."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "neural_tts.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    cli()
