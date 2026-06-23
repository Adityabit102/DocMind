"""Application configuration.

A single ``Settings`` object (Pydantic ``BaseSettings``) reads every tunable
value from environment variables / ``.env``. Defaults are *local-first*: with no
API keys present the app falls back to HuggingFace embeddings + Ollama so it runs
end-to-end at zero cost. Imported everywhere via the cached ``get_settings()``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["openai", "anthropic", "ollama", "groq"]
EmbeddingProvider = Literal["openai", "huggingface"]
RetrievalMode = Literal["similarity", "mmr", "hybrid"]
ChunkStrategy = Literal["recursive", "semantic", "parent"]
CacheBackend = Literal["memory", "redis"]
RerankerProvider = Literal["cross_encoder", "cohere", "voyage"]


class Settings(BaseSettings):
    """Type-validated application settings, sourced from env / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────
    llm_provider: LLMProvider = "ollama"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    llm_max_tokens: int = Field(default=1024, ge=128, le=8192)

    # ── Embeddings ───────────────────────────────────────────────────
    embedding_provider: EmbeddingProvider = "huggingface"
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── Retrieval ────────────────────────────────────────────────────
    chunk_size: int = Field(default=1000, ge=200, le=4000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    chunk_strategy: ChunkStrategy = "recursive"
    retrieval_k: int = Field(default=20, ge=1, le=50)
    reranker_top_k: int = Field(default=5, ge=1, le=20)
    retrieval_mode: RetrievalMode = "hybrid"
    ensemble_dense_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    ensemble_bm25_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    reranker_provider: RerankerProvider = "cross_encoder"
    cohere_api_key: str = ""
    voyage_api_key: str = ""
    enable_query_expansion: bool = True
    enable_hyde: bool = False
    enable_reranking: bool = True
    enable_context_compression: bool = False
    enable_crag: bool = True
    enable_self_rag: bool = False

    # ── Cache ────────────────────────────────────────────────────────
    cache_backend: CacheBackend = "memory"
    redis_url: str = "redis://localhost:6379"
    cache_ttl_seconds: int = 86_400
    cache_score_threshold: float = Field(default=0.95, ge=0.0, le=1.0)

    # ── Observability ────────────────────────────────────────────────
    langsmith_api_key: str = ""
    langchain_tracing_v2: bool = False
    langchain_project: str = "docmind"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    enable_query_logging: bool = True

    # ── Security ─────────────────────────────────────────────────────
    api_key_auth_enabled: bool = False
    api_key: str = "your-secret-key"
    # User accounts + per-user document/conversation isolation. Off by default so
    # the local-first single-user path needs no login.
    enable_auth: bool = False
    auth_secret: str = "change-me-in-production"
    auth_token_ttl_hours: int = Field(default=168, ge=1, le=8760)
    users_file: str = "data/users.json"
    cors_origins: str = "http://localhost:3000,http://localhost:8501"
    api_rate_limit: int = 60
    max_upload_size_mb: int = 50
    max_query_chars: int = 2000
    enable_pii_redaction: bool = False
    enable_prompt_guard: bool = False

    # ── Storage ──────────────────────────────────────────────────────
    upload_dir: str = "data/uploads"
    faiss_index_dir: str = "data/faiss_index"
    data_dir: str = "data"
    # Persist the data/ dir to a (private) HuggingFace dataset so it survives
    # ephemeral hosts like HF Spaces. Both must be set to enable sync.
    hf_token: str = ""
    hf_dataset_repo: str = ""  # e.g. "username/docmind-data"
    eval_results_dir: str = "evaluation/results"
    eval_testset_dir: str = "evaluation/testsets"
    metadata_file: str = "data/metadata.json"
    log_file: str = "logs/app.log"

    # ── Evaluation ───────────────────────────────────────────────────
    ragas_test_size: int = 50
    drift_alert_threshold: float = 0.15
    enable_scheduled_eval: bool = False
    scheduled_eval_interval_hours: int = Field(default=168, ge=1, le=8760)

    # ── Ingestion (vision) ───────────────────────────────────────────
    enable_image_captioning: bool = False
    vision_model: str = "llava"

    # ── Derived / helpers ────────────────────────────────────────────
    @field_validator("cors_origins")
    @classmethod
    def _strip_origins(cls, v: str) -> str:
        return ",".join(o.strip() for o in v.split(",") if o.strip())

    @property
    def cors_origin_list(self) -> list[str]:
        return [o for o in self.cors_origins.split(",") if o]

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def has_anthropic(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def has_groq(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def has_hf_sync(self) -> bool:
        return bool(self.hf_token and self.hf_dataset_repo)

    @property
    def effective_embedding_provider(self) -> EmbeddingProvider:
        """Local-first: silently fall back to HuggingFace if OpenAI is selected
        without a key, so indexing never hard-fails on a missing secret."""
        if self.embedding_provider == "openai" and not self.has_openai:
            return "huggingface"
        return self.embedding_provider

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached ``Settings`` instance."""
    return Settings()


def persist_settings_to_env(updates: dict[str, object], env_path: str = ".env") -> None:
    """Write settings ``updates`` back to ``.env`` so they survive a restart.

    Keys are upper-cased to match the env-var convention; existing lines are
    rewritten in place and new keys appended. Pydantic reads case-insensitively,
    so this round-trips cleanly into the next ``Settings()`` load.
    """
    import os

    lines: list[str] = []
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()

    remaining = {k.upper(): v for k, v in updates.items()}
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            out.append(line)
            continue
        key = stripped.split("=", 1)[0].strip().upper()
        if key in remaining:
            out.append(f"{key}={remaining.pop(key)}")
        else:
            out.append(line)
    for key, value in remaining.items():
        out.append(f"{key}={value}")

    with open(env_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")


settings = get_settings()
