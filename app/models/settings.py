"""Pydantic schemas for the runtime-settings endpoints (GET / PATCH)."""

from __future__ import annotations

from pydantic import BaseModel


class SettingsResponse(BaseModel):
    """Current effective configuration exposed to the UI sidebar."""

    llm_provider: str
    llm_model: str
    embedding_provider: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    chunk_strategy: str
    retrieval_k: int
    reranker_top_k: int
    retrieval_mode: str
    enable_query_expansion: bool
    enable_hyde: bool
    enable_reranking: bool
    enable_context_compression: bool
    enable_crag: bool
    enable_self_rag: bool
    cache_backend: str
    cache_ttl_seconds: int
    llm_temperature: float
    llm_max_tokens: int
    log_level: str
    enable_scheduled_eval: bool
    scheduled_eval_interval_hours: int
    enable_image_captioning: bool
    vision_model: str
    reranker_provider: str
    enable_prompt_guard: bool
    enable_auth: bool


class SettingsPatch(BaseModel):
    """Partial settings update — any subset of mutable fields."""

    llm_provider: str | None = None
    llm_model: str | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    chunk_strategy: str | None = None
    retrieval_k: int | None = None
    reranker_top_k: int | None = None
    retrieval_mode: str | None = None
    enable_query_expansion: bool | None = None
    enable_hyde: bool | None = None
    enable_reranking: bool | None = None
    enable_context_compression: bool | None = None
    enable_crag: bool | None = None
    enable_self_rag: bool | None = None
    cache_backend: str | None = None
    cache_ttl_seconds: int | None = None
    llm_temperature: float | None = None
    llm_max_tokens: int | None = None
    log_level: str | None = None
    enable_scheduled_eval: bool | None = None
    scheduled_eval_interval_hours: int | None = None
    enable_image_captioning: bool | None = None
    vision_model: str | None = None
    reranker_provider: str | None = None
    enable_prompt_guard: bool | None = None
    enable_auth: bool | None = None
