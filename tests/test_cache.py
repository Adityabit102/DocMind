"""Cost-tracking + cache helper tests."""

from __future__ import annotations

from rag.generation.cost_tracker import approx_tokens, estimate_cost


def test_estimate_cost_known_model():
    breakdown = estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000)
    # 0.15 in + 0.60 out per 1M tokens
    assert round(breakdown.cost_usd, 2) == 0.75


def test_estimate_cost_local_is_free():
    breakdown = estimate_cost("llama3", 5000, 5000)
    assert breakdown.cost_usd == 0.0


def test_approx_tokens():
    assert approx_tokens("") == 1
    assert approx_tokens("a" * 400) == 100
