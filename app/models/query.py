"""Pydantic schemas for queries, answers, and feedback."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class RetrievalMode(str, Enum):
    SIMILARITY = "similarity"
    MMR = "mmr"
    HYBRID = "hybrid"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SourceChunk(BaseModel):
    """A retrieved chunk attached to an answer as a citation."""

    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str = ""
    filename: str = ""
    page_number: int = 0
    chunk_index: int = 0
    char_offset: int = 0
    text: str = ""
    relevance_score: float = 0.0
    reranker_score: float | None = None
    embedding_model: str = ""


class QueryRequest(BaseModel):
    """Inbound Q&A request. Per-query overrides default settings when provided."""

    question: str = Field(..., min_length=1, max_length=8000)
    session_id: str | None = None
    # Prior turns for multi-turn context: [{"role": "user"|"assistant", "content": ...}].
    chat_history: list[dict[str, str]] | None = None
    retrieval_mode: RetrievalMode | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    document_ids: list[str] | None = None
    tags: list[str] | None = None
    page_range: tuple[int, int] | None = None
    enable_query_expansion: bool | None = None
    enable_hyde: bool | None = None
    enable_reranking: bool | None = None
    enable_context_compression: bool | None = None
    enable_cache: bool | None = None


class QueryResponse(BaseModel):
    """Full (non-streamed) answer payload."""

    query_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str = ""
    answer: str = ""
    sources: list[SourceChunk] = Field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    confidence_score: float = 0.0
    grounding_score: float | None = None
    sentence_support: list[dict] = Field(default_factory=list)
    retrieval_mode: str = "hybrid"
    llm_model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    cache_hit: bool = False
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RegenerateRequest(QueryRequest):
    """Regenerate an existing answer with a different strategy."""

    query_id: str


class FeedbackRequest(BaseModel):
    """Thumbs up/down feedback on an answer."""

    query_id: str
    helpful: bool
    comment: str | None = None


class FollowupRequest(BaseModel):
    """Ask the model to suggest follow-up questions for an answered query."""

    question: str = Field(..., min_length=1, max_length=8000)
    answer: str = Field(..., min_length=1, max_length=8000)
