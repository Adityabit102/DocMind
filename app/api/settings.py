"""Settings endpoints: read effective config, patch it, clear cache."""

from __future__ import annotations

from fastapi import APIRouter

from app.config import get_settings, persist_settings_to_env
from app.models.settings import SettingsPatch, SettingsResponse

router = APIRouter(prefix="/api/v1", tags=["settings"])


def _to_response() -> SettingsResponse:
    s = get_settings()
    return SettingsResponse(
        llm_provider=s.llm_provider,
        llm_model=s.llm_model,
        embedding_provider=s.effective_embedding_provider,
        embedding_model=s.embedding_model,
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        chunk_strategy=s.chunk_strategy,
        retrieval_k=s.retrieval_k,
        reranker_top_k=s.reranker_top_k,
        retrieval_mode=s.retrieval_mode,
        enable_query_expansion=s.enable_query_expansion,
        enable_hyde=s.enable_hyde,
        enable_reranking=s.enable_reranking,
        enable_context_compression=s.enable_context_compression,
        enable_crag=s.enable_crag,
        enable_self_rag=s.enable_self_rag,
        cache_backend=s.cache_backend,
        cache_ttl_seconds=s.cache_ttl_seconds,
        llm_temperature=s.llm_temperature,
        llm_max_tokens=s.llm_max_tokens,
        log_level=s.log_level,
        enable_scheduled_eval=s.enable_scheduled_eval,
        scheduled_eval_interval_hours=s.scheduled_eval_interval_hours,
        enable_image_captioning=s.enable_image_captioning,
        vision_model=s.vision_model,
        reranker_provider=s.reranker_provider,
        enable_prompt_guard=s.enable_prompt_guard,
        enable_auth=s.enable_auth,
    )


@router.get("/settings", response_model=SettingsResponse)
def read_settings() -> SettingsResponse:
    return _to_response()


@router.patch("/settings", response_model=SettingsResponse)
def update_settings(patch: SettingsPatch) -> SettingsResponse:
    """Apply a runtime settings patch to the in-process config.

    Mutates the cached ``Settings`` so the change takes effect immediately for
    subsequent requests, then persists the patch to ``.env`` so it survives a
    restart.
    """
    s = get_settings()
    applied = patch.model_dump(exclude_none=True)
    for field, value in applied.items():
        if hasattr(s, field):
            setattr(s, field, value)
    # Embeddings/LLM factories are cached — clear so new settings take effect.
    from rag.generation.llm_factory import get_llm
    from rag.ingestion.embedder import get_embeddings

    get_embeddings.cache_clear()
    get_llm.cache_clear()
    if applied:
        persist_settings_to_env(applied)
    return _to_response()


@router.delete("/cache")
def clear_semantic_cache() -> dict[str, str]:
    from rag.cache.semantic_cache import clear_cache

    clear_cache()
    return {"status": "cache cleared"}
