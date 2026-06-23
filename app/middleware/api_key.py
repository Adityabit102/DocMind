"""Optional API-key authentication via the ``X-API-Key`` header.

Disabled by default (local dev). When ``API_KEY_AUTH_ENABLED`` is true, every
request except open paths (health, docs, metrics) must carry the configured key.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.config import get_settings

_OPEN_PATHS = {"/api/v1/health", "/metrics", "/docs", "/redoc", "/openapi.json", "/"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
        settings = get_settings()
        if not settings.api_key_auth_enabled:
            return await call_next(request)
        if request.url.path in _OPEN_PATHS or request.method == "OPTIONS":
            return await call_next(request)
        provided = request.headers.get("X-API-Key", "")
        if provided != settings.api_key:
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})
        return await call_next(request)
