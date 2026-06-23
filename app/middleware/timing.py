"""Inject an ``X-Process-Time`` header with per-request latency."""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time"] = f"{(time.perf_counter() - start) * 1000:.2f}ms"
        return response
