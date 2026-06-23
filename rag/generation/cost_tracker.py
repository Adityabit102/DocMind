"""Token → USD cost estimation per provider/model.

Prices are USD per 1M tokens (input, output), accurate to 2025 list pricing.
Unknown models and local providers (Ollama) cost zero. Used to surface per-query
and per-session spend in the UI and query logs.
"""

from __future__ import annotations

from dataclasses import dataclass

# (input_per_1m, output_per_1m) in USD.
_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-3.5-turbo": (0.50, 1.50),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-haiku-20240307": (0.25, 1.25),
}


@dataclass
class CostBreakdown:
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> CostBreakdown:
    """Compute USD cost for a single call. Local/unknown models → $0."""
    in_price, out_price = _PRICES.get(model, (0.0, 0.0))
    cost = (prompt_tokens / 1_000_000) * in_price + (completion_tokens / 1_000_000) * out_price
    return CostBreakdown(prompt_tokens, completion_tokens, round(cost, 8))


def approx_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) when no usage metadata is returned."""
    return max(1, len(text) // 4)
