"""Synthetic Q&A test-set generation for evaluation.

Prefers RAGAS ``TestsetGenerator`` (needs an OpenAI key). With no key it falls
back to a lightweight heuristic generator so evaluation still runs in local-first
mode. Test sets persist to ``evaluation/testsets/``.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone

from app.config import get_settings


def _persist(rows: list[dict[str, str]]) -> str:
    out_dir = get_settings().eval_testset_dir
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"testset_{stamp}.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["question", "ground_truth"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def _ragas_testset(documents, test_size: int) -> list[dict[str, str]]:  # type: ignore[no-untyped-def]
    from ragas.testset.generator import TestsetGenerator

    generator = TestsetGenerator.with_openai()
    testset = generator.generate_with_langchain_docs(documents, test_size=test_size)
    df = testset.to_pandas()
    return [
        {"question": r["question"], "ground_truth": r.get("ground_truth", "")}
        for _, r in df.iterrows()
    ]


def _heuristic_testset(store, test_size: int) -> list[dict[str, str]]:  # type: ignore[no-untyped-def]
    """No-key fallback: turn chunk leads into simple 'what does X say' probes."""
    rows: list[dict[str, str]] = []
    for chunk in store.chunks[:test_size]:
        snippet = chunk.page_content.strip().split(".")[0][:160]
        if not snippet:
            continue
        rows.append(
            {
                "question": f"What does the document say about: {snippet}?",
                "ground_truth": snippet,
            }
        )
    return rows


def generate_testset(store=None, test_size: int | None = None) -> list[dict[str, str]]:  # type: ignore[no-untyped-def]
    """Generate and persist a synthetic test set; return its rows."""
    settings = get_settings()
    test_size = test_size or settings.ragas_test_size

    if store is None:
        from rag.ingestion.indexer import IndexStore

        store = IndexStore().load()
    if store.is_empty:
        return []

    try:
        if settings.has_openai:
            rows = _ragas_testset(store.chunks, test_size)
        else:
            rows = _heuristic_testset(store, test_size)
    except Exception:  # noqa: BLE001 — RAGAS failure → heuristic fallback
        rows = _heuristic_testset(store, test_size)

    if rows:
        _persist(rows)
    return rows
