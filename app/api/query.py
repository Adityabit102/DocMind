"""Query endpoints: blocking Q&A, SSE streaming, regenerate, feedback."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.config import Settings, get_settings
from app.dependencies import AppState, get_state
from app.models.conversation import Message, Role
from app.models.query import (
    FeedbackRequest,
    FollowupRequest,
    QueryRequest,
    QueryResponse,
    RegenerateRequest,
    SourceChunk,
)
from rag.generation.chain import RAGChain
from rag.generation.llm_factory import build_llm
from rag.generation.memory import ConversationMemory
from rag.retrieval.factory import RetrievalOverrides
from rag.security.pii_redactor import redact

router = APIRouter(prefix="/api/v1", tags=["query"])


def _memory_from_history(history: list[dict[str, str]] | None) -> ConversationMemory | None:
    """Build conversation memory from client-supplied prior turns.

    The chat UI keeps history locally, so it sends the recent turns with each
    request; we fold consecutive (user, assistant) pairs into memory the chain
    injects into the prompt, making follow-up questions context-aware.
    """
    if not history:
        return None
    memory = ConversationMemory()
    pending_user: str | None = None
    for msg in history:
        role, content = msg.get("role"), msg.get("content", "")
        if role == "user":
            pending_user = content
        elif role == "assistant" and pending_user is not None:
            memory.add_turn(pending_user, content)
            pending_user = None
    return memory if memory.turns else None


def _memory_for(session_id: str | None, state: AppState) -> ConversationMemory | None:
    """Build conversation memory from a stored session's prior turns, if any."""
    if not session_id or session_id not in state.conversations:
        return None
    memory = ConversationMemory()
    messages = state.conversations[session_id].messages
    pending_user: str | None = None
    for msg in messages:
        if msg.role == Role.USER:
            pending_user = msg.content
        elif msg.role == Role.ASSISTANT and pending_user is not None:
            memory.add_turn(pending_user, msg.content)
            pending_user = None
    return memory


def _record_turn(
    session_id: str | None,
    question: str,
    response: QueryResponse,
    memory: ConversationMemory | None,
    llm: object,
    state: AppState,
) -> None:
    """Append the user/assistant turn to the session and roll up older context."""
    if not session_id or session_id not in state.conversations:
        return
    session = state.conversations[session_id]
    session.messages.append(Message(role=Role.USER, content=question))
    session.messages.append(
        Message(role=Role.ASSISTANT, content=response.answer, sources=response.sources)
    )
    session.total_cost_usd += response.cost_usd
    session.total_tokens += response.prompt_tokens + response.completion_tokens
    # Fold turns beyond the window into a rolling summary to bound prompt size.
    if memory is not None:
        memory.add_turn(question, response.answer)
        try:
            memory.summarise_old(llm)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001 — summarisation is best-effort
            pass


def _overrides(req: QueryRequest) -> RetrievalOverrides:
    return RetrievalOverrides(
        mode=req.retrieval_mode.value if req.retrieval_mode else None,
        top_k=req.top_k,
        enable_query_expansion=req.enable_query_expansion,
        enable_hyde=req.enable_hyde,
        enable_context_compression=req.enable_context_compression,
        document_ids=req.document_ids,
        tags=req.tags,
        page_range=req.page_range,
    )


def _validate(req: QueryRequest, settings: Settings) -> None:
    if len(req.question) > settings.max_query_chars:
        raise HTTPException(status_code=400, detail="Query exceeds maximum length")
    if settings.enable_prompt_guard:
        from rag.security.guardrail import screen_query

        verdict = screen_query(req.question)
        if verdict.blocked:
            raise HTTPException(status_code=400, detail=verdict.message)


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, settings: Settings = Depends(get_settings)) -> QueryResponse:
    """Run a single grounded Q&A query and return the answer with citations."""
    _validate(req, settings)
    state = get_state()
    if state.store.is_empty:
        raise HTTPException(status_code=409, detail="No documents indexed yet")

    llm = build_llm(settings, streaming=False)
    chain = RAGChain(state.store, llm, settings)
    memory = _memory_from_history(req.chat_history) or _memory_for(req.session_id, state)
    result = chain.answer(req.question, _overrides(req), memory)
    answer = redact(result.answer)

    response = QueryResponse(
        session_id=req.session_id or "",
        answer=answer,
        sources=[SourceChunk(**s) for s in result.sources],
        confidence=result.confidence,
        confidence_score=result.confidence_score,
        grounding_score=result.grounding_score,
        sentence_support=result.sentence_support,
        retrieval_mode=result.retrieval_mode,
        llm_model=result.llm_model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
    )
    _record_turn(req.session_id, req.question, response, memory, llm, state)
    if settings.enable_query_logging:
        state.query_logs.append(json.loads(response.model_dump_json()))
    return response


@router.post("/query/stream")
async def query_stream(req: QueryRequest, settings: Settings = Depends(get_settings)):  # type: ignore[no-untyped-def]
    """Stream a grounded answer token-by-token over SSE, ending with citations."""
    _validate(req, settings)
    state = get_state()
    if state.store.is_empty:
        raise HTTPException(status_code=409, detail="No documents indexed yet")

    import time

    start = time.perf_counter()
    chain = RAGChain(state.store, build_llm(settings, streaming=True), settings)
    memory = _memory_from_history(req.chat_history) or _memory_for(req.session_id, state)
    token_iter, sources, meta = chain.stream(req.question, _overrides(req), memory)

    async def event_generator():  # type: ignore[no-untyped-def]
        from rag.evaluation.grounding import sentence_support

        collected: list[str] = []
        for chunk in token_iter:
            text = getattr(chunk, "content", None) or (chunk if isinstance(chunk, str) else "")
            if text:
                collected.append(text)
                yield {"event": "token", "data": json.dumps({"token": text})}
        # Grounding is computed post-generation against the cited chunks.
        answer = "".join(collected)
        spans = sentence_support(answer, [s.get("text", "") for s in sources])
        scored = [s for s in spans if s["text"].strip()]
        grounding = (
            round(sum(1 for s in scored if s["supported"]) / len(scored), 4)
            if scored
            else None
        )
        # Token/cost estimate for the per-message footer (streaming has no usage
        # callback, so approximate from the text).
        from rag.generation.cost_tracker import approx_tokens, estimate_cost

        context_text = " ".join(s.get("text", "") for s in sources)
        prompt_tokens = approx_tokens(context_text + req.question)
        completion_tokens = approx_tokens(answer)
        cost = estimate_cost(meta.llm_model, prompt_tokens, completion_tokens)
        # Log streamed queries too, so the analytics panel reflects chat usage
        # (only the blocking endpoint logged before).
        if settings.enable_query_logging:
            state.query_logs.append(
                {
                    "answer": answer,
                    "confidence": meta.confidence.value,
                    "confidence_score": meta.confidence_score,
                    "retrieval_mode": meta.retrieval_mode,
                    "llm_model": meta.llm_model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cost_usd": cost.cost_usd,
                    "latency_ms": int((time.perf_counter() - start) * 1000),
                }
            )
        yield {
            "event": "done",
            "data": json.dumps(
                {
                    "sources": sources,
                    "meta": {
                        "confidence": meta.confidence.value,
                        "confidence_score": meta.confidence_score,
                        "grounding_score": grounding,
                        "sentence_support": spans,
                        "model": meta.llm_model,
                        "retrieval_mode": meta.retrieval_mode,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "cost_usd": cost.cost_usd,
                    },
                }
            ),
        }

    return EventSourceResponse(event_generator())


@router.post("/query/regenerate", response_model=QueryResponse)
def regenerate(req: RegenerateRequest, settings: Settings = Depends(get_settings)) -> QueryResponse:
    """Regenerate an answer, typically with a different retrieval strategy."""
    return query(req, settings)


@router.post("/feedback")
def feedback(req: FeedbackRequest) -> dict[str, str]:
    """Record thumbs up/down feedback for an answer."""
    state = get_state()
    state.feedback.append(req.model_dump())
    return {"status": "recorded", "query_id": req.query_id}


@router.post("/query/followups")
def followups(req: FollowupRequest, settings: Settings = Depends(get_settings)) -> dict[str, list[str]]:
    """Suggest a few natural follow-up questions for the answered query."""
    prompt = (
        "Given the question and the answer below, suggest exactly 3 concise "
        "follow-up questions a curious reader might ask next. Return only the "
        "questions, one per line, with no numbering or extra text.\n\n"
        f"Question: {req.question}\n\nAnswer: {req.answer}\n\nFollow-up questions:"
    )
    try:
        result = build_llm(settings, streaming=False).invoke(prompt)
        text = getattr(result, "content", str(result))
    except Exception:  # noqa: BLE001 — suggestions are best-effort
        return {"followups": []}
    lines = [
        line.strip().lstrip("0123456789.-) ").strip()
        for line in text.splitlines()
        if line.strip()
    ]
    questions = [line for line in lines if line.endswith("?")][:3]
    return {"followups": questions}
