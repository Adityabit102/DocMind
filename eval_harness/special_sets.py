"""Beyond-clean QA sets: distribution shift and adversarial-lite inputs.

These are the "not just clean validation data" part of the harness (design doc
§5.3), adapted to the reasoning-based extractive-QA task. Both reuse
``data_prep._answer_for`` so ground truth is computed identically to training.

distribution_shift_set:
    QA over an **unseen "statement" layout** the model was never trained on
    (training uses invoice / purchase_order / quote). Same values (dates, line
    items, totals), different surface form — tests whether the reasoning
    (latest-date, max-price, count) transfers off the training layouts.

adversarial_lite_set:
    Normal layouts, but questions phrased to **lure the model toward a
    distractor** — e.g. asking for the most expensive line item with wording
    that echoes the (larger) total, or the latest date while naming the fields.
"""

from __future__ import annotations

import random

from finetune.data_prep import (
    QUESTION_TYPES,
    _answer_for,
    make_fields,
    render_document,
)


def _statement_layout(fx) -> str:
    """An unseen 'statement' layout carrying the same values as ``fx``."""
    d0, d1, d2 = fx.dates
    items = "; ".join(f"{li.name} ({li.qty} @ {li.price_str})" for li in fx.line_items)
    return (f"ACCOUNT STATEMENT — ref {fx.doc_number}\n"
            f"Issued by {fx.sender} to {fx.recipient}.\n"
            f"Key dates on file: {d0}, {d1}, {d2}.\n"
            f"Line items: {items}.\n"
            f"Net {fx.subtotal}; balance carried {fx.total}.")


def distribution_shift_set(n_docs: int = 18, seed: int = 101) -> list[dict]:
    rng = random.Random(seed)
    from finetune.data_prep import _QUESTIONS

    rows: list[dict] = []
    for i in range(n_docs):
        fx = make_fields(rng, rng.choice(["invoice", "purchase_order", "quote"]))
        ctx = _statement_layout(fx)
        for qtype in QUESTION_TYPES:
            rows.append({"context": ctx, "question": rng.choice(_QUESTIONS[qtype]),
                         "answer": _answer_for(fx, qtype), "qtype": qtype,
                         "group_id": f"shift-{i}", "split": "shift"})
    rng.shuffle(rows)
    return rows


# Misleading phrasings that nudge toward a distractor.
_ADV_QUESTIONS = {
    "total": "What is the grand total to pay, after subtotal, tax/shipping and discounts?",
    "sender": "Which party issued or prepared this — the company, not the person it's addressed to?",
    "latest_date": "Reading top to bottom, ignore document order — which date is chronologically the latest?",
    "earliest_date": "Reading top to bottom, ignore document order — which date is chronologically the earliest?",
    "max_lineitem": "What is the single highest item price — an individual line item, NOT the subtotal or total?",
    "num_items": "Counting only the itemised products (not totals or dates), how many line items are there?",
    "priciest_item": "Which product line is the costliest — give the item name, not its price?",
}


def adversarial_lite_set(n_docs: int = 18, seed: int = 202) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for i in range(n_docs):
        fx = make_fields(rng, rng.choice(["invoice", "purchase_order", "quote"]))
        ctx = render_document(rng, fx)
        for qtype in QUESTION_TYPES:
            rows.append({"context": ctx, "question": _ADV_QUESTIONS[qtype],
                         "answer": _answer_for(fx, qtype), "qtype": qtype,
                         "group_id": f"adv-{i}", "split": "adversarial"})
    rng.shuffle(rows)
    return rows
