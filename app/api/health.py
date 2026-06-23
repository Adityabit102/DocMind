"""Health probe and Prometheus metrics endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.responses import Response

from app.dependencies import get_state

router = APIRouter()

# ── Prometheus metrics (module-level singletons) ─────────────────────
QUERY_COUNTER = Counter(
    "rag_queries_total", "Total queries", ["model", "cache_hit", "retrieval_mode"]
)
QUERY_LATENCY = Histogram("rag_query_latency_seconds", "Query latency (s)")
DOCUMENTS_GAUGE = Gauge("rag_documents_total", "Indexed document count")
CHUNKS_GAUGE = Gauge("rag_chunks_total", "Indexed chunk count")
INGEST_DURATION = Histogram("rag_ingestion_duration_seconds", "Ingestion duration (s)")
CACHE_HIT_RATIO = Gauge("rag_cache_hit_ratio", "Cache hit ratio")
ERRORS_COUNTER = Counter("rag_errors_total", "Total errors", ["error_type"])


@router.get("/api/v1/health")
def health() -> dict[str, str]:
    """Liveness probe used by the Docker health check."""
    return {"status": "ok", "service": "docmind"}


@router.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics (refreshes gauges from current state)."""
    state = get_state()
    CHUNKS_GAUGE.set(state.store.chunk_count)
    DOCUMENTS_GAUGE.set(len(state.store.document_ids()))
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/api/v1/admin/stats")
def admin_stats(state=Depends(get_state)) -> dict[str, int]:  # type: ignore[no-untyped-def]
    """Storage / document / chunk / query counts for the admin panel."""
    return {
        "document_count": len(state.store.document_ids()),
        "chunk_count": state.store.chunk_count,
        "query_count": len(state.query_logs),
        "conversation_count": len(state.conversations),
        "feedback_count": len(state.feedback),
    }


@router.get("/api/v1/admin/analytics")
def admin_analytics(state=Depends(get_state)) -> dict:  # type: ignore[no-untyped-def, type-arg]
    """Aggregate query telemetry for the analytics panel."""
    logs = state.query_logs
    n = len(logs)

    def _avg(key: str) -> float:
        vals = [float(x.get(key, 0) or 0) for x in logs]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    feedback = state.feedback
    helpful = sum(1 for f in feedback if f.get("helpful"))
    recent = [
        {
            "answer": (x.get("answer", "") or "")[:80],
            "confidence": x.get("confidence", ""),
            "latency_ms": x.get("latency_ms", 0),
            "cost_usd": x.get("cost_usd", 0),
            "model": x.get("llm_model", ""),
            "timestamp": x.get("timestamp", ""),
        }
        for x in logs[-10:][::-1]
    ]
    return {
        "query_count": n,
        "avg_latency_ms": _avg("latency_ms"),
        "total_cost_usd": round(sum(float(x.get("cost_usd", 0) or 0) for x in logs), 6),
        "total_tokens": int(
            sum(int(x.get("prompt_tokens", 0) or 0) + int(x.get("completion_tokens", 0) or 0) for x in logs)
        ),
        "avg_confidence_score": _avg("confidence_score"),
        "feedback_count": len(feedback),
        "helpful_count": helpful,
        "helpful_ratio": round(helpful / len(feedback), 3) if feedback else 0.0,
        "recent": recent,
    }
