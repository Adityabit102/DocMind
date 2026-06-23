"""The end-to-end RAG chain: retrieve → (CRAG) → rerank → prompt → LLM → answer.

Wraps the retrieval factory and generation pieces into one object exposing
``answer`` (blocking) and ``stream`` (token generator). Reranking, CRAG, and
Self-RAG are toggled by settings/overrides; the chain reports the scores and
citations the API surfaces to the UI.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.config import Settings, get_settings
from app.models.query import ConfidenceLevel
from rag.evaluation.grounding import sentence_support
from rag.generation.cost_tracker import approx_tokens, estimate_cost
from rag.generation.memory import ConversationMemory
from rag.generation.self_rag import reflect
from rag.ingestion.indexer import IndexStore
from rag.retrieval.factory import RetrievalOverrides, build_retriever, filter_documents
from rag.retrieval.parent_expansion import expand_to_parents
from rag.retrieval.query_expansion import hyde_query, rewrite_query
from rag.retrieval.reranker import deduplicate, rerank

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "prompts")
_NO_CONTEXT = "I don't have enough information in the provided documents to answer that."
# Below this top reranker relevance (sigmoid 0..1), CRAG rewrites and retries.
_CRAG_THRESHOLD = 0.2


def _load_prompt(name: str) -> str:
    with open(os.path.join(_PROMPTS_DIR, name), encoding="utf-8") as fh:
        return fh.read()


def format_docs_with_citations(docs: list[Document]) -> str:
    """Render retrieved chunks as ``[Source: file, Page: N] text`` blocks."""
    blocks = []
    for doc in docs:
        src = doc.metadata.get("filename", doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page_number", "?")
        blocks.append(f"[Source: {src}, Page: {page}] {doc.page_content}")
    return "\n\n".join(blocks)


def _confidence(score: float) -> ConfidenceLevel:
    # ``score`` is the top reranker relevance probability (sigmoid of the
    # cross-encoder logit). Calibrated to that distribution: strong matches sit
    # around 0.5+, plausible ones around 0.25.
    if score >= 0.5:
        return ConfidenceLevel.HIGH
    if score >= 0.25:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


def _to_source_dicts(scored: list[tuple[Document, float]]) -> list[dict[str, Any]]:
    sources = []
    for doc, score in scored:
        sources.append(
            {
                "document_id": doc.metadata.get("document_id", ""),
                "filename": doc.metadata.get("filename", doc.metadata.get("source", "")),
                "page_number": doc.metadata.get("page_number", 0),
                "chunk_index": doc.metadata.get("chunk_index", 0),
                "char_offset": doc.metadata.get("char_offset", 0),
                "text": doc.page_content,
                "relevance_score": round(float(score), 4),
                "reranker_score": round(float(score), 4),
            }
        )
    return sources


@dataclass
class GenerationResult:
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    confidence_score: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    llm_model: str = ""
    retrieval_mode: str = "hybrid"
    grounding_score: float | None = None
    sentence_support: list[dict[str, Any]] = field(default_factory=list)


class RAGChain:
    """Composable RAG pipeline over an ``IndexStore``."""

    def __init__(
        self,
        store: IndexStore,
        llm: BaseLanguageModel,
        settings: Settings | None = None,
    ):
        self.store = store
        self.llm = llm
        self.settings = settings or get_settings()
        self._system = _load_prompt("system_default.txt")
        self._qa = _load_prompt("rag_qa.txt")

    # ── Retrieval + reranking shared by answer() and stream() ────────
    def _prepare(
        self, query: str, overrides: RetrievalOverrides, memory: ConversationMemory | None
    ) -> tuple[list[tuple[Document, float]], str]:
        s = self.settings
        search_query = query
        if s.enable_query_expansion and memory and memory.turns:
            search_query = rewrite_query(query, self.llm, memory.as_text())

        # Document/tag scoping is applied *after* retrieval, so widen the
        # candidate pool when it is active — otherwise a small top-k can be
        # filled entirely by out-of-scope chunks and leave nothing in scope.
        retr_overrides = overrides
        if overrides.document_ids or overrides.tags:
            from dataclasses import replace

            pool = max(overrides.top_k or s.retrieval_k, 30)
            retr_overrides = replace(overrides, top_k=pool)
        retriever = build_retriever(self.store, self.llm, retr_overrides, s)
        # HyDE: retrieve against a synthesized hypothetical answer passage.
        use_hyde = overrides.enable_hyde if overrides.enable_hyde is not None else s.enable_hyde
        if use_hyde:
            docs = retriever.invoke(hyde_query(search_query, self.llm))
        else:
            docs = retriever.invoke(search_query)
        docs = filter_documents(docs, overrides)
        docs = deduplicate(docs)
        # Parent-document expansion: swap matched child chunks for their parents.
        if s.chunk_strategy == "parent":
            docs = expand_to_parents(docs, self.store)

        top_k = overrides.top_k or s.reranker_top_k

        def _rank(candidates: list[Document]) -> list[tuple[Document, float]]:
            # Rerank the *full* candidate set once; CRAG and the final top-k both
            # read from this single pass (the cross-encoder scores every pair
            # regardless of top_k, so a second call would just redo the work).
            if s.enable_reranking and candidates:
                return rerank(query, candidates, top_k=len(candidates))
            return [(d, 1.0 / (i + 1)) for i, d in enumerate(candidates)]

        scored_all = _rank(docs)
        # CRAG: if even the best reranked chunk is weak, rewrite the query and
        # retrieve once more. Uses the already-computed top score — no extra rerank.
        if s.enable_crag and s.enable_reranking and scored_all and scored_all[0][1] < _CRAG_THRESHOLD:
            refined = deduplicate(
                filter_documents(retriever.invoke(rewrite_query(query, self.llm, "")), overrides)
            )
            if s.chunk_strategy == "parent":
                refined = expand_to_parents(refined, self.store)
            if refined:
                scored_all = _rank(refined)
        scored = scored_all[:top_k]
        return scored, search_query

    def _build_prompt(self, memory: ConversationMemory | None):  # type: ignore[no-untyped-def]
        messages: list[tuple[str, str]] = [("system", self._system)]
        if memory:
            for user, assistant in memory.recent():
                messages.append(("human", user))
                messages.append(("ai", assistant))
        messages.append(("human", self._qa))
        return ChatPromptTemplate.from_messages(messages)

    # ── Public API ───────────────────────────────────────────────────
    def answer(
        self,
        query: str,
        overrides: RetrievalOverrides | None = None,
        memory: ConversationMemory | None = None,
    ) -> GenerationResult:
        """Run the full pipeline and return a complete answer with citations."""
        start = time.perf_counter()
        overrides = overrides or RetrievalOverrides()
        scored, _ = self._prepare(query, overrides, memory)

        if not scored:  # zero-chunk guardrail — never call the LLM blind
            return GenerationResult(
                answer=_NO_CONTEXT,
                confidence=ConfidenceLevel.LOW,
                latency_ms=int((time.perf_counter() - start) * 1000),
                llm_model=self.settings.llm_model,
            )

        docs = [d for d, _ in scored]
        context = format_docs_with_citations(docs)
        prompt = self._build_prompt(memory)
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": query})

        # Self-RAG: reflect on grounding; abstain if the answer isn't supported.
        grounding_score: float | None = None
        if self.settings.enable_self_rag:
            verdict = reflect(answer, context, self.llm)
            grounding_score = round(verdict.score, 4)
            if not verdict.grounded and verdict.score < 0.5:
                answer = _NO_CONTEXT

        # Per-sentence grounding (cheap, lexical) for hallucination highlighting.
        chunk_texts = [d.page_content for d in docs]
        spans = sentence_support(answer, chunk_texts)
        if grounding_score is None:
            supported = [s for s in spans if s["text"].strip()]
            grounding_score = (
                round(sum(1 for s in supported if s["supported"]) / len(supported), 4)
                if supported
                else None
            )

        top_score = max(s for _, s in scored)
        prompt_tokens = approx_tokens(context + query + self._system)
        completion_tokens = approx_tokens(answer)
        cost = estimate_cost(self.settings.llm_model, prompt_tokens, completion_tokens)
        return GenerationResult(
            answer=answer,
            sources=_to_source_dicts(scored),
            confidence=_confidence(top_score),
            confidence_score=round(float(top_score), 4),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost.cost_usd,
            latency_ms=int((time.perf_counter() - start) * 1000),
            llm_model=self.settings.llm_model,
            retrieval_mode=overrides.mode or self.settings.retrieval_mode,
            grounding_score=grounding_score,
            sentence_support=spans,
        )

    def stream(
        self,
        query: str,
        overrides: RetrievalOverrides | None = None,
        memory: ConversationMemory | None = None,
    ) -> tuple[Iterator[Any], list[dict[str, Any]], GenerationResult]:
        """Return (token_iterator, sources, partial_result) for SSE streaming.

        The iterator yields LangChain message chunks; ``partial_result`` carries
        the metadata (confidence, cost estimate) known before generation.
        """
        overrides = overrides or RetrievalOverrides()
        scored, _ = self._prepare(query, overrides, memory)
        if not scored:
            def _empty() -> Iterator[Any]:
                yield _NO_CONTEXT
            return _empty(), [], GenerationResult(
                answer="", confidence=ConfidenceLevel.LOW, llm_model=self.settings.llm_model
            )

        docs = [d for d, _ in scored]
        context = format_docs_with_citations(docs)
        prompt = self._build_prompt(memory)
        chain = prompt | self.llm
        token_iter = chain.stream({"context": context, "question": query})
        top_score = max(s for _, s in scored)
        meta = GenerationResult(
            answer="",
            sources=_to_source_dicts(scored),
            confidence=_confidence(top_score),
            confidence_score=round(float(top_score), 4),
            llm_model=self.settings.llm_model,
            retrieval_mode=overrides.mode or self.settings.retrieval_mode,
        )
        return token_iter, _to_source_dicts(scored), meta
