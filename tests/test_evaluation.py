"""Evaluation, security, and access-control tests."""

from __future__ import annotations

from langchain_core.documents import Document

from app.models.document import Classification
from rag.evaluation.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from rag.security.access_control import can_access, security_trim
from rag.security.pii_redactor import contains_pii, redact


def test_recall_and_ndcg_edge_cases():
    assert recall_at_k([], set(), 5) == 0.0
    assert reciprocal_rank(["a", "b"], {"b"}) == 0.5
    assert ndcg_at_k(["a"], set(), 1) == 0.0


def test_access_control_levels():
    assert can_access("admin", Classification.CONFIDENTIAL)
    assert not can_access("user", Classification.CONFIDENTIAL)
    assert can_access("guest", Classification.PUBLIC)


def test_security_trim_filters_confidential():
    docs = [
        Document(page_content="public", metadata={"classification": "public"}),
        Document(page_content="secret", metadata={"classification": "confidential"}),
    ]
    trimmed = security_trim(docs, "user")
    assert len(trimmed) == 1
    assert trimmed[0].page_content == "public"


def test_pii_regex_redaction():
    text = "Contact me at john.doe@example.com or 555-123-4567."
    assert contains_pii(text)
    cleaned = redact(text, force=True)
    assert "john.doe@example.com" not in cleaned
    assert "REDACTED" in cleaned
