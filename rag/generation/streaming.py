"""SSE helpers for token-by-token streaming of answers.

Formats LangChain stream chunks as Server-Sent Events the FastAPI
``StreamingResponse`` (and the Next.js EventSource client) consume. Each token is
a ``data:`` event; a terminal ``event: done`` frame carries the source citations.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any


def sse_event(data: str, event: str | None = None) -> str:
    """Encode one SSE frame."""
    prefix = f"event: {event}\n" if event else ""
    payload = data.replace("\n", "\ndata: ")
    return f"{prefix}data: {payload}\n\n"


def sse_token(token: str) -> str:
    """A single answer token as a JSON-wrapped SSE frame."""
    return sse_event(json.dumps({"token": token}), event="token")


def sse_done(sources: list[dict[str, Any]], meta: dict[str, Any] | None = None) -> str:
    """Terminal frame: citations + metadata (cost, confidence, latency)."""
    body = {"sources": sources, "meta": meta or {}}
    return sse_event(json.dumps(body), event="done")


def sse_error(message: str) -> str:
    return sse_event(json.dumps({"error": message}), event="error")


async def stream_text(chunks: Iterator[Any]) -> AsyncIterator[str]:
    """Adapt a sync LangChain token stream into async SSE token frames."""
    for chunk in chunks:
        text = getattr(chunk, "content", None)
        if text is None:
            text = str(chunk)
        if text:
            yield sse_token(text)
