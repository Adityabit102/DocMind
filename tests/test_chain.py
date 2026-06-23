"""Generation tests: citation formatting, confidence, conversation memory."""

from __future__ import annotations

from app.models.query import ConfidenceLevel
from rag.generation.chain import _confidence, format_docs_with_citations
from rag.generation.memory import ConversationMemory


def test_format_docs_with_citations(sample_docs):
    rendered = format_docs_with_citations(sample_docs)
    assert "[Source: report.pdf, Page: 1]" in rendered
    assert "[Source: study.pdf, Page: 1]" in rendered


def test_confidence_thresholds():
    # Calibrated to the reranker's sigmoid relevance probability: >=0.5 high,
    # >=0.25 medium, else low.
    assert _confidence(0.9) == ConfidenceLevel.HIGH
    assert _confidence(0.5) == ConfidenceLevel.HIGH
    assert _confidence(0.35) == ConfidenceLevel.MEDIUM
    assert _confidence(0.1) == ConfidenceLevel.LOW


def test_memory_window_and_text():
    mem = ConversationMemory(window=2)
    mem.add_turn("q1", "a1")
    mem.add_turn("q2", "a2")
    mem.add_turn("q3", "a3")
    recent = mem.recent()
    assert len(recent) == 2
    assert recent[0] == ("q2", "a2")
    assert "q3" in mem.as_text()


def test_memory_as_messages(sample_docs):
    mem = ConversationMemory()
    mem.add_turn("hello", "hi there")
    messages = mem.as_messages()
    assert len(messages) == 2
    assert messages[0].content == "hello"


def test_memory_clear():
    mem = ConversationMemory()
    mem.add_turn("a", "b")
    mem.clear()
    assert mem.turns == []
