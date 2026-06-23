"""LLM provider factory with a graceful fallback chain.

Selection follows ``LLM_PROVIDER`` but degrades automatically: a provider that
needs an absent API key is skipped in favour of the next available one, ending at
local Ollama which needs no key. This keeps the app answerable in local-first mode.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.language_models import BaseLanguageModel

from app.config import Settings, get_settings


def _build_openai(settings: Settings, streaming: bool) -> BaseLanguageModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        streaming=streaming,
        api_key=settings.openai_api_key,
    )


def _build_groq(settings: Settings, streaming: bool) -> BaseLanguageModel:
    """Groq via its OpenAI-compatible endpoint — fast, free-tier hosted Llama/etc."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        streaming=streaming,
        api_key=settings.groq_api_key,
        base_url=settings.groq_base_url,
    )


def _build_anthropic(settings: Settings, streaming: bool) -> BaseLanguageModel:
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        streaming=streaming,
        api_key=settings.anthropic_api_key,
    )


def _build_ollama(settings: Settings, streaming: bool) -> BaseLanguageModel:
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        base_url=settings.ollama_base_url,
    )


def _resolve_provider(settings: Settings) -> str:
    """Pick the first usable provider, honouring availability of API keys."""
    chain = [settings.llm_provider, "groq", "openai", "anthropic", "ollama"]
    for provider in chain:
        if provider == "groq" and settings.has_groq:
            return "groq"
        if provider == "openai" and settings.has_openai:
            return "openai"
        if provider == "anthropic" and settings.has_anthropic:
            return "anthropic"
        if provider == "ollama":
            return "ollama"
    return "ollama"


def build_llm(settings: Settings | None = None, streaming: bool = False) -> BaseLanguageModel:
    """Construct the effective chat LLM for the resolved provider."""
    settings = settings or get_settings()
    provider = _resolve_provider(settings)
    if provider == "groq":
        return _build_groq(settings, streaming)
    if provider == "openai":
        return _build_openai(settings, streaming)
    if provider == "anthropic":
        return _build_anthropic(settings, streaming)
    return _build_ollama(settings, streaming)


@lru_cache
def get_llm() -> BaseLanguageModel:
    """Process-wide cached non-streaming LLM."""
    return build_llm()
