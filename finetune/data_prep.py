"""Dataset preparation for the LoRA document extractive-QA fine-tune.

WHY THIS TASK (and why it replaced document-type classification):
An earlier version of this module framed the fine-tune as 6-class document-type
classification. That task **saturated at 1.00 clean accuracy** — a TF-IDF +
logistic-regression baseline also scored 1.000/1.000 macro-F1, confirming it is
a trivially-separable keyword task that under-tests the model. So the task was
replaced with **DocVQA-style extractive question answering**: given a document
and a question, the model must locate and copy the exact answer span.

What makes it genuinely hard (and keeps clean accuracy well below 1.00):
  - Every document contains **distractors** — multiple dates (issue date, due
    date, delivery date), multiple money amounts (subtotal, tax, total),
    multiple person/company names (from, to, prepared-by). The model must
    disambiguate *which* value the question asks for, not just recognise a type.
  - Answers are exact spans; scoring is exact-match (EM) + token-level F1
    (SQuAD-style), so partial/adjacent errors are penalised.

This is still offline-reproducible: documents are synthesised from seeded
templates, and every (question, answer) pair is generated from the *known*
field values, so ground truth is exact and unambiguous.

Split methodology (unchanged discipline):
  - GROUPED by document instance: all QA pairs from one document land in exactly
    one split — a paraphrased/degraded twin can never straddle train/test.
  - STRATIFIED by question type so each answer category is represented in each
    split. Ratios 70/15/15, seeded.

Leakage check (unchanged, runs before any result may be claimed):
  - Exact-duplicate check on the (normalised) CONTEXT via SHA-1 — a context must
    not appear in more than one split (hard failure).
  - Near-duplicate context check (token-set Jaccard >= 0.9) across splits,
    counted and reported.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

# Question types. Two are simple lookups (anchors); the rest require REASONING
# over multiple in-document values — comparison, aggregation, or counting — so
# the model cannot solve the task with a "find the matching label, copy the next
# token" shortcut. (An earlier lookup-only version of this QA task saturated at
# EM 1.00; these reasoning questions are what make it genuinely hard.)
QUESTION_TYPES = [
    "total",            # lookup (anchor)
    "sender",           # lookup (anchor)
    "latest_date",      # reasoning: max over the 3 dates
    "earliest_date",    # reasoning: min over the 3 dates
    "max_lineitem",     # reasoning: max price across line items (NOT the total)
    "num_items",        # reasoning: count of line items
    "priciest_item",    # reasoning: name of the most expensive line item
]

# The instruction wrapper for the causal-LM SFT. The model emits the answer
# span after "Answer:". Eval parses the same shape.
PROMPT_TEMPLATE = (
    "Extract the answer to the question from the document. "
    "Reply with the exact value only.\n\n"
    "Document:\n{doc}\n\nQuestion: {question}\nAnswer:"
)


def build_prompt(doc: str, question: str) -> str:
    return PROMPT_TEMPLATE.format(doc=doc.strip(), question=question.strip())


# ── Seeded content banks ─────────────────────────────────────────────────
_FIRST = ["Ava", "Liam", "Noah", "Mia", "Priya", "Kenji", "Sofia", "Omar",
          "Elena", "Marcus", "Aisha", "Diego", "Hana", "Tariq", "Lena", "Ravi"]
_LAST = ["Osei", "Nakamura", "Delgado", "Kowalski", "Abebe", "Rossi", "Haddad",
         "Fischer", "Okonkwo", "Petrov", "Silva", "Ahmed", "Novak", "Reyes"]
_COMPANY = ["Northwind Traders", "Acme Logistics", "Blue Harbor Foods",
            "Meridian Analytics", "Cedar & Vale LLP", "Orbital Systems",
            "Sunfield Grocers", "Ironbridge Consulting", "Halcyon Media"]
_ITEM = ["Wireless mouse", "USB-C hub", "Standing desk", "Notebook (pack of 5)",
         "Ergonomic chair", "Monitor arm", "Coffee beans 1kg", "Whiteboard"]
# Three document "styles" so distribution-shift can hold one out cleanly.
DOC_STYLES = ["invoice", "purchase_order", "quote"]


def _money(rng: random.Random) -> str:
    return f"${rng.randint(50, 9999)}.{rng.randint(0, 99):02d}"


def _date(rng: random.Random) -> str:
    return f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"


@dataclass
class LineItem:
    name: str
    qty: int
    price_val: float
    price_str: str


@dataclass
class DocFields:
    style: str
    doc_number: str
    dates: list[str]       # [issue, due, delivery] — all distinct, chronologically mixed
    subtotal: str
    total: str
    sender: str            # issuing company
    recipient: str         # person billed / shipped to
    line_items: list[LineItem]


def _distinct_dates(rng: random.Random, n: int) -> list[str]:
    out: set[str] = set()
    while len(out) < n:
        out.add(_date(rng))
    return list(out)


def _distinct_prices(rng: random.Random, n: int) -> list[float]:
    out: set[float] = set()
    while len(out) < n:
        out.add(round(rng.uniform(20, 990), 2))
    return list(out)


def make_fields(rng: random.Random, style: str) -> DocFields:
    n_items = rng.randint(2, 5)
    prices = _distinct_prices(rng, n_items)
    items = [LineItem(name=rng.choice(_ITEM), qty=rng.randint(1, 6),
                      price_val=p, price_str=f"${p:.2f}") for p in prices]
    return DocFields(
        style=style,
        doc_number=f"{ {'invoice':'INV','purchase_order':'PO','quote':'QT'}[style] }-{rng.randint(10000, 99999)}",
        dates=_distinct_dates(rng, 3),
        subtotal=_money(rng),
        total=_money(rng),
        sender=rng.choice(_COMPANY),
        recipient=f"{rng.choice(_FIRST)} {rng.choice(_LAST)}",
        line_items=items,
    )


def render_document(rng: random.Random, fx: DocFields) -> str:
    """Render a document embedding all values PLUS distractors. The three dates
    are printed in a FIXED positional order (issue, due, delivery) but are NOT
    chronologically ordered, so 'latest/earliest date' cannot be read off by
    position — it requires actually comparing them."""
    issue, due, delivery = fx.dates
    items = "\n".join(f"  {li.name}  x{li.qty}   {li.price_str}" for li in fx.line_items)
    tax = _money(rng)  # decoy money amount
    if fx.style == "invoice":
        return (f"INVOICE  {fx.doc_number}\nFrom: {fx.sender}\nBill to: {fx.recipient}\n"
                f"Issue date: {issue}   Due date: {due}\n"
                f"Delivery: {delivery}\nItems:\n{items}\n"
                f"Subtotal: {fx.subtotal}\nTax: {tax}\nAmount due: {fx.total}\n"
                "Payment terms: Net 30.")
    if fx.style == "purchase_order":
        return (f"PURCHASE ORDER  {fx.doc_number}\nVendor: {fx.sender}\nShip to: {fx.recipient}\n"
                f"Order date: {issue}   Required by: {due}\n"
                f"Requested delivery: {delivery}\nItems:\n{items}\n"
                f"Subtotal: {fx.subtotal}\nShipping: {tax}\nOrder total: {fx.total}\n"
                "Please confirm receipt.")
    # quote
    return (f"QUOTATION  {fx.doc_number}\nPrepared by: {fx.sender}\nFor: {fx.recipient}\n"
            f"Quote date: {issue}   Expires: {due}\n"
            f"Estimated delivery: {delivery}\nItems:\n{items}\n"
            f"Subtotal: {fx.subtotal}\nDiscount: {tax}\nEstimated total: {fx.total}\n"
            "This is not an invoice.")


# Question phrasings per type (varied so the model can't key on one string).
# The reasoning types deliberately have NO matching field label in the document.
_QUESTIONS: dict[str, list[str]] = {
    "total": ["What is the total amount?", "What is the amount due / order total?",
              "What is the final total?"],
    "sender": ["Who issued this document?", "Which company is the sender / vendor?"],
    "latest_date": ["Which of the dates on this document is the latest (most recent)?",
                    "Of all dates shown, which is the latest calendar date?"],
    "earliest_date": ["Which of the dates on this document is the earliest?",
                      "Of all dates shown, which is the earliest calendar date?"],
    "max_lineitem": ["What is the price of the most expensive line item (not the total)?",
                     "Among the individual line items, what is the highest item price?"],
    "num_items": ["How many line items are listed?", "How many distinct items are on this document?"],
    "priciest_item": ["Which line item is the most expensive?",
                      "What is the name of the highest-priced item?"],
}


def _answer_for(fx: DocFields, qtype: str) -> str:
    if qtype == "total":
        return fx.total
    if qtype == "sender":
        return fx.sender
    if qtype == "latest_date":
        return max(fx.dates)          # ISO yyyy-mm-dd -> lexical == chronological
    if qtype == "earliest_date":
        return min(fx.dates)
    if qtype == "num_items":
        return str(len(fx.line_items))
    priciest = max(fx.line_items, key=lambda li: li.price_val)
    if qtype == "max_lineitem":
        return priciest.price_str
    if qtype == "priciest_item":
        return priciest.name
    raise KeyError(qtype)


@dataclass
class QAExample:
    context: str
    question: str
    answer: str
    qtype: str
    group_id: str  # document instance
    split: str = ""


def generate_corpus(n_docs: int, seed: int) -> list[QAExample]:
    """Deterministically synthesise QA pairs (all 7 question types per document)."""
    rng = random.Random(seed)
    examples: list[QAExample] = []
    for i in range(n_docs):
        style = DOC_STYLES[i % len(DOC_STYLES)]
        fx = make_fields(rng, style)
        doc = render_document(rng, fx)
        gid = f"{style}-{i:04d}"
        for qtype in QUESTION_TYPES:
            question = rng.choice(_QUESTIONS[qtype])
            examples.append(QAExample(context=doc, question=question,
                                      answer=_answer_for(fx, qtype),
                                      qtype=qtype, group_id=gid))
    rng.shuffle(examples)
    return examples


# ── Normalisation + hashing for the leakage check ────────────────────────
def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _sha1(text: str) -> str:
    return hashlib.sha1(_normalise(text).encode("utf-8")).hexdigest()


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ── Grouped, stratified split ────────────────────────────────────────────
def split_dataset(examples: list[QAExample],
                  ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
                  seed: int = 13) -> list[QAExample]:
    """Grouped by document; whole documents go to one split. Stratified by style."""
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
    rng = random.Random(seed)
    # Group -> style, and group -> examples.
    groups: dict[str, list[QAExample]] = defaultdict(list)
    group_style: dict[str, str] = {}
    for ex in examples:
        groups[ex.group_id].append(ex)
        group_style[ex.group_id] = ex.group_id.split("-")[0]

    by_style: dict[str, list[str]] = defaultdict(list)
    for gid, style in group_style.items():
        by_style[style].append(gid)

    for style, gids in by_style.items():
        rng.shuffle(gids)
        n = len(gids)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        for j, gid in enumerate(gids):
            split = "train" if j < n_train else "val" if j < n_train + n_val else "test"
            for ex in groups[gid]:
                ex.split = split
    return examples


@dataclass
class LeakageReport:
    exact_overlaps: int
    near_dupe_pairs: int
    per_split_counts: dict
    per_qtype_counts: dict
    passed: bool
    details: list = field(default_factory=list)


def leakage_check(examples: list[QAExample], jaccard_threshold: float = 0.9) -> LeakageReport:
    """Fail on exact cross-split CONTEXT duplicates; count near-dupes as a warning."""
    by_split: dict[str, list[QAExample]] = defaultdict(list)
    for ex in examples:
        by_split[ex.split].append(ex)

    # Exact-duplicate check on unique contexts (dedupe by group to avoid counting
    # the 7 QA rows per doc as 7 "duplicates").
    ctx_by_group: dict[str, tuple[str, str]] = {}
    for ex in examples:
        ctx_by_group[ex.group_id] = (ex.split, ex.context)
    hash_to_splits: dict[str, set[str]] = defaultdict(set)
    for split, ctx in ctx_by_group.values():
        hash_to_splits[_sha1(ctx)].add(split)
    exact = sum(1 for splits in hash_to_splits.values() if len(splits) > 1)

    # Near-duplicate contexts across splits (unique contexts per split).
    train_ctx = list({ctx for (s, ctx) in ctx_by_group.values() if s == "train"})
    train_tokens = [(_token_set(c), c) for c in train_ctx]
    near = 0
    details: list = []
    for other in ("val", "test"):
        other_ctx = list({ctx for (s, ctx) in ctx_by_group.values() if s == other})
        for c in other_ctx:
            ts = _token_set(c)
            for tt, _tc in train_tokens:
                if _jaccard(ts, tt) >= jaccard_threshold:
                    near += 1
                    if len(details) < 10:
                        details.append(f"near-dupe context train ~ {other}")
                    break

    per_split = {k: len(v) for k, v in sorted(by_split.items())}
    per_qtype = {split: dict(Counter(e.qtype for e in rows))
                 for split, rows in sorted(by_split.items())}
    return LeakageReport(exact_overlaps=exact, near_dupe_pairs=near,
                         per_split_counts=per_split, per_qtype_counts=per_qtype,
                         passed=(exact == 0), details=details)


# ── Public entrypoint ────────────────────────────────────────────────────
def build_dataset(n_docs: int = 120, seed: int = 42,
                  out_dir: Path | None = None) -> tuple[dict[str, list[QAExample]], LeakageReport]:
    out_dir = Path(out_dir) if out_dir else DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = generate_corpus(n_docs=n_docs, seed=seed)
    examples = split_dataset(examples, seed=seed + 1)
    report = leakage_check(examples)
    if not report.passed:
        raise RuntimeError(
            f"Data leakage detected: {report.exact_overlaps} exact cross-split "
            "context duplicates. Refusing to proceed — fix the generator/split first."
        )

    splits: dict[str, list[QAExample]] = defaultdict(list)
    for ex in examples:
        splits[ex.split].append(ex)

    for split, rows in splits.items():
        with (out_dir / f"{split}.jsonl").open("w") as f:
            for ex in rows:
                f.write(json.dumps({
                    "context": ex.context, "question": ex.question, "answer": ex.answer,
                    "qtype": ex.qtype, "group_id": ex.group_id, "split": ex.split,
                }) + "\n")

    card = {
        "task": "document extractive QA (7 question types, with in-document distractors)",
        "question_types": QUESTION_TYPES,
        "doc_styles": DOC_STYLES,
        "generation": {"n_docs": n_docs, "qa_pairs": len(examples), "seed": seed,
                       "synthetic_templated": True},
        "difficulty": "each doc embeds decoy dates/amounts/names so the model must "
                      "disambiguate which value the question targets",
        "split_methodology": "grouped-by-document + stratified-by-style, 70/15/15",
        "metric": "exact-match (EM) + token-level F1 (SQuAD-style)",
        "leakage_check": {
            "exact_cross_split_context_overlaps": report.exact_overlaps,
            "near_duplicate_context_pairs_ge_0.9_jaccard": report.near_dupe_pairs,
            "passed": report.passed,
        },
        "counts": {"per_split": report.per_split_counts, "per_qtype": report.per_qtype_counts},
    }
    (out_dir / "dataset_card.json").write_text(json.dumps(card, indent=2))
    return dict(splits), report


def load_splits(data_dir: Path | None = None) -> dict[str, list[dict]]:
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    if not (data_dir / "train.jsonl").exists():
        build_dataset(out_dir=data_dir)
    out: dict[str, list[dict]] = {}
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}.jsonl"
        out[split] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return out


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Build + leakage-check the extractive-QA dataset")
    ap.add_argument("--n-docs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _, rep = build_dataset(n_docs=args.n_docs, seed=args.seed)
    print("Dataset written to", DATA_DIR)
    print("Per-split counts:", rep.per_split_counts)
    print("Exact cross-split context overlaps:", rep.exact_overlaps, "(0 required)")
    print("Near-duplicate context pairs (>=0.9 Jaccard):", rep.near_dupe_pairs)
    print("Leakage check passed:", rep.passed)
