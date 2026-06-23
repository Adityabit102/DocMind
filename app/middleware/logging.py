"""Structured JSON request/response logging middleware + logger setup."""

from __future__ import annotations

import logging
import time

from pythonjsonlogger import jsonlogger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.config import get_settings


def configure_logging() -> None:
    """Install a JSON formatter on the root logger + file handler."""
    settings = get_settings()
    fmt = jsonlogger.JsonFormatter("%(asctime)s %(name)s %(levelname)s %(message)s")

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    handlers: list[logging.Handler] = [stream]

    try:
        import os

        os.makedirs(os.path.dirname(settings.log_file) or ".", exist_ok=True)
        file_handler = logging.FileHandler(settings.log_file)
        file_handler.setFormatter(fmt)
        handlers.append(file_handler)
    except OSError:
        pass

    root = logging.getLogger()
    root.handlers = handlers
    root.setLevel(settings.log_level)


_SECRET_KEYS = {"api_key", "openai_api_key", "anthropic_api_key", "authorization"}


def scrub_secrets(data: dict) -> dict:
    """Mask known secret keys before logging."""
    return {k: ("***" if k.lower() in _SECRET_KEYS else v) for k, v in data.items()}


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log method, path, status, and latency for every request as JSON."""

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
        logger = logging.getLogger("docmind.request")
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": round(elapsed_ms, 2),
            },
        )
        return response
