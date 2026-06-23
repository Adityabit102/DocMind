"""Pydantic schemas for evaluation runs and drift detection."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field


class EvaluationRequest(BaseModel):
    """Trigger a RAGAS evaluation run."""

    test_size: int = Field(default=50, ge=1, le=500)
    document_ids: list[str] | None = None
    regenerate_testset: bool = False


class PerQuestionResult(BaseModel):
    """RAGAS scores for a single synthetic question."""

    question: str
    answer: str = ""
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0


class EvaluationResult(BaseModel):
    """Aggregate result of one RAGAS evaluation run."""

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    engine: str = "lexical"  # "ragas" | "llm-judge:<model>" | "lexical"
    question_count: int = 0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    unsupported_sentence_ratio: float = 0.0
    recall_at_k: float = 0.0
    ndcg: float = 0.0
    mrr: float = 0.0
    per_question: list[PerQuestionResult] = Field(default_factory=list)


class ABArmConfig(BaseModel):
    """One retrieval configuration under test in an A/B comparison."""

    label: str = "arm"
    retrieval_mode: str | None = None
    top_k: int | None = None
    enable_query_expansion: bool | None = None
    enable_hyde: bool | None = None
    enable_context_compression: bool | None = None


class ABArmScore(BaseModel):
    """Aggregate scorecard for one A/B arm."""

    label: str
    mean_grounding: float = 0.0
    mean_usr: float = 0.0
    mean_confidence: float = 0.0
    mean_latency_ms: float = 0.0


class ABTestRequest(BaseModel):
    """Compare two retrieval configurations over a shared question set."""

    test_size: int = Field(default=10, ge=1, le=100)
    arm_a: ABArmConfig = Field(default_factory=lambda: ABArmConfig(label="A"))
    arm_b: ABArmConfig = Field(default_factory=lambda: ABArmConfig(label="B"))


class ABTestResult(BaseModel):
    """Result of an A/B retrieval comparison with per-metric deltas."""

    question_count: int = 0
    arm_a: ABArmScore
    arm_b: ABArmScore
    winner: str = "tie"
    deltas: dict[str, float] = Field(default_factory=dict)


class DriftStatus(BaseModel):
    """Embedding-distribution drift snapshot."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    drift_detected: bool = False
    drift_score: float = 0.0
    threshold: float = 0.15
    reference_window_days: int = 7
    message: str = "OK"
