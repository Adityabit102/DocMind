"""DocMind FastAPI application entry point.

Wires configuration, structured logging, middleware (CORS, timing, request
logging, optional API-key auth, rate limiting), all routers, and lifespan events
(load the index + install the semantic cache on startup).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api import auth, conversations, documents, evaluate, health, query
from app.api import settings as settings_api
from app.config import get_settings
from app.dependencies import init_state
from app.middleware.api_key import APIKeyMiddleware
from app.middleware.logging import RequestLoggingMiddleware, configure_logging
from app.middleware.rate_limiter import limiter
from app.middleware.timing import TimingMiddleware

logger = logging.getLogger("docmind")


def _setup_langsmith() -> None:
    """Enable LangSmith tracing when configured (key + tracing flag present).

    LangChain reads these via environment variables, so we export them before
    any chain runs. No-op when no key is set — keeps the local-first default.
    """
    import os

    s = get_settings()
    if s.langchain_tracing_v2 and s.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = s.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = s.langchain_project
        logger.info("LangSmith tracing enabled (project=%s)", s.langchain_project)


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    """Startup: configure logging, load the index, install the semantic cache."""
    configure_logging()
    _setup_langsmith()
    # Restore persisted state (FAISS index, uploads, registry) before loading it.
    try:
        from rag.persistence import hf_sync

        hf_sync.pull()
    except Exception as exc:  # noqa: BLE001 — persistence is best-effort
        logger.warning("Data restore unavailable: %s", exc)
    init_state()
    try:
        from app.dependencies import get_store
        from rag.ingestion.jobs import start_worker

        start_worker(get_store)
    except Exception as exc:  # noqa: BLE001 — queue worker is best-effort
        logger.warning("Ingestion worker unavailable: %s", exc)
    try:
        from rag.cache.semantic_cache import setup_semantic_cache

        backend = setup_semantic_cache()
        logger.info("Semantic cache active (%s)", backend)
    except Exception as exc:  # noqa: BLE001 — cache is best-effort
        logger.warning("Semantic cache unavailable: %s", exc)
    try:
        from rag.evaluation.scheduler import start_scheduler

        start_scheduler()
    except Exception as exc:  # noqa: BLE001 — scheduler is best-effort
        logger.warning("Scheduled evaluation unavailable: %s", exc)
    logger.info("DocMind backend started")
    yield
    try:
        from rag.evaluation.scheduler import stop_scheduler

        stop_scheduler()
    except Exception:  # noqa: BLE001
        pass
    try:
        from rag.persistence import hf_sync

        hf_sync.push("shutdown")
    except Exception:  # noqa: BLE001 — best-effort final flush
        pass
    logger.info("DocMind backend shutting down")


def create_app() -> FastAPI:
    """Application factory."""
    settings = get_settings()
    app = FastAPI(
        title="DocMind API",
        description="AI-Powered RAG Document Q&A System",
        version="1.0.0",
        lifespan=lifespan,
    )

    # SlowAPI rate limiting.
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Middleware (outermost first).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(APIKeyMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(TimingMiddleware)

    # Routers.
    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(documents.router)
    app.include_router(query.router)
    app.include_router(conversations.router)
    app.include_router(evaluate.router)
    app.include_router(settings_api.router)

    @app.get("/", tags=["root"])
    def root() -> dict[str, str]:
        return {"service": "DocMind", "docs": "/docs", "health": "/api/v1/health"}

    return app


app = create_app()
