"""Shared application state and FastAPI dependency providers.

A single ``AppState`` holds the loaded index, in-memory conversation registry,
query logs, and feedback. Provider functions are wired into routes via
``Depends`` so handlers stay thin and testable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings, get_settings
from app.models.conversation import ConversationSession
from rag.ingestion.indexer import IndexStore


@dataclass
class AppState:
    """Process-wide mutable state, created once at startup."""

    store: IndexStore
    conversations: dict[str, ConversationSession] = field(default_factory=dict)
    query_logs: list[dict[str, Any]] = field(default_factory=list)
    feedback: list[dict[str, Any]] = field(default_factory=list)
    cache_backend: str = "memory"


_state: AppState | None = None


def init_state() -> AppState:
    """Build and cache the app state (called from the FastAPI lifespan)."""
    global _state
    store = IndexStore().load()
    # Prune any index chunks orphaned from the registry so retrieval stays
    # consistent with what the documents API reports.
    try:
        from rag.ingestion.pipeline import reconcile_index

        removed = reconcile_index(store)
        if removed:
            logging.getLogger("docmind").info("Reconciled index: pruned %d orphan chunks", removed)
    except Exception:  # noqa: BLE001 — reconciliation is best-effort
        pass
    _state = AppState(store=store)
    return _state


def get_state() -> AppState:
    """Return the initialised app state (lazy-init for tests/standalone use)."""
    if _state is None:
        return init_state()
    return _state


# ── FastAPI providers ────────────────────────────────────────────────
def get_store() -> IndexStore:
    return get_state().store


def settings_dep() -> Settings:
    return get_settings()


def get_chain():  # type: ignore[no-untyped-def]
    """Build a fresh RAG chain bound to the current store + LLM."""
    from rag.generation.chain import RAGChain
    from rag.generation.llm_factory import build_llm

    state = get_state()
    return RAGChain(state.store, build_llm(streaming=False))
