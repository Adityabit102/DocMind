"""Semantic LLM cache: in-memory for dev, Redis for production.

Wires LangChain's global LLM cache so semantically similar queries return a
cached completion (cosine ≥ threshold) without a fresh LLM call. Redis failures
degrade gracefully to the in-memory backend.
"""

from __future__ import annotations

import logging

from app.config import Settings, get_settings
from rag.ingestion.embedder import get_embeddings

logger = logging.getLogger(__name__)


def _set_llm_cache(cache) -> None:  # type: ignore[no-untyped-def]
    """Version-resilient ``set_llm_cache`` (relocated in langchain >= 1.x)."""
    try:
        from langchain.globals import set_llm_cache
    except ImportError:  # pragma: no cover - version shim
        from langchain_core.globals import set_llm_cache
    set_llm_cache(cache)


def _in_memory_cache():  # type: ignore[no-untyped-def]
    try:
        from langchain_community.cache import InMemoryCache
    except ImportError:  # pragma: no cover - version shim
        from langchain_core.caches import InMemoryCache
    return InMemoryCache()


def setup_semantic_cache(settings: Settings | None = None) -> str:
    """Install the configured semantic cache globally. Returns the active backend."""
    settings = settings or get_settings()

    if settings.cache_backend == "redis":
        try:
            from langchain_community.cache import RedisSemanticCache

            _set_llm_cache(
                RedisSemanticCache(
                    embedding=get_embeddings(),  # only the semantic backend needs embeddings
                    redis_url=settings.redis_url,
                    score_threshold=settings.cache_score_threshold,
                )
            )
            return "redis"
        except Exception as exc:  # noqa: BLE001 — fall back, never crash on cache
            logger.warning("Redis cache unavailable (%s); using in-memory cache", exc)

    _set_llm_cache(_in_memory_cache())
    return "memory"


def clear_cache() -> None:
    """Drop all cached completions (admin action)."""
    _set_llm_cache(_in_memory_cache())
