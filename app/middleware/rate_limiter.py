"""Per-client rate limiting backed by SlowAPI (token bucket per IP)."""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import get_settings


def build_limiter() -> Limiter:
    """Create a SlowAPI limiter with the configured per-minute default."""
    settings = get_settings()
    return Limiter(
        key_func=get_remote_address,
        default_limits=[f"{settings.api_rate_limit}/minute"],
    )


limiter = build_limiter()
