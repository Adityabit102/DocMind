"""Task metrics + a reusable QA-model wrapper for the LoRA extractive-QA model.

The fine-tune is framed as extractive QA: given a document + question, the
causal LM emits the answer span after "Answer:". This module provides:

  - `normalize_answer` / `exact_match` / `token_f1` — SQuAD-style scoring, so
    formatting differences don't count as errors but wrong values do.
  - `QAModel` — loads base model (+ optional LoRA adapter) and turns
    (context, question) into a predicted answer. Reused by the trainer's
    validation metric, this CLI, and the degradation harness so all three score
    the model identically.
  - `qa_metrics` — mean EM + mean token-F1, overall and per question type.
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from finetune.data_prep import QUESTION_TYPES, build_prompt, load_splits


# ── SQuAD-style normalisation + scoring ──────────────────────────────────
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    p_toks = normalize_answer(pred).split()
    g_toks = normalize_answer(gold).split()
    if not p_toks and not g_toks:
        return 1.0
    if not p_toks or not g_toks:
        return 0.0
    common = Counter(p_toks) & Counter(g_toks)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(p_toks)
    recall = same / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def parse_answer(generated: str) -> str:
    """Take the model's answer text: first non-empty line, trimmed."""
    text = generated.strip()
    # Stop at the first newline / an injected new 'Question:'/'Document:' block.
    for stop in ("\nQuestion:", "\nDocument:", "\n\n"):
        idx = text.find(stop)
        if idx != -1:
            text = text[:idx]
    first = text.splitlines()[0] if text.splitlines() else text
    return first.strip().strip(".").strip()


@dataclass
class QAModel:
    """Thin inference wrapper: (context, question) -> predicted answer span."""

    base_model: str
    adapter_dir: str | None = None
    device: str = "cpu"
    max_new_tokens: int = 16
    _tok: object = field(default=None, repr=False)
    _model: object = field(default=None, repr=False)

    def load(self) -> "QAModel":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.base_model)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(self.base_model, torch_dtype=torch.float32)
        if self.adapter_dir:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_dir)
            model = model.merge_and_unload()
        self._model = model.to(self.device).eval()
        return self

    def predict(self, context: str, question: str) -> str:
        import torch

        prompt = build_prompt(context, question)
        enc = self._tok(prompt, return_tensors="pt", truncation=True, max_length=640)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self._model.generate(
                **enc, max_new_tokens=self.max_new_tokens,
                do_sample=False, num_beams=1, pad_token_id=self._tok.pad_token_id,
            )
        gen = self._tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return parse_answer(gen)

    def predict_many(self, pairs: list[tuple[str, str]]) -> list[str]:
        return [self.predict(c, q) for c, q in pairs]


@dataclass
class Metrics:
    exact_match: float
    token_f1: float
    support: int
    per_qtype_em: dict = field(default_factory=dict)
    per_qtype_f1: dict = field(default_factory=dict)


def qa_metrics(rows: list[dict], preds: list[str]) -> Metrics:
    ems = [exact_match(p, r["answer"]) for p, r in zip(preds, rows)]
    f1s = [token_f1(p, r["answer"]) for p, r in zip(preds, rows)]
    by_em: dict[str, list[float]] = {}
    by_f1: dict[str, list[float]] = {}
    for r, em, f1 in zip(rows, ems, f1s):
        by_em.setdefault(r["qtype"], []).append(em)
        by_f1.setdefault(r["qtype"], []).append(f1)
    mean = lambda xs: sum(xs) / max(len(xs), 1)
    return Metrics(
        exact_match=mean(ems), token_f1=mean(f1s), support=len(rows),
        per_qtype_em={q: round(mean(by_em.get(q, [])), 3) for q in QUESTION_TYPES},
        per_qtype_f1={q: round(mean(by_f1.get(q, [])), 3) for q in QUESTION_TYPES},
    )


def evaluate_split(model: QAModel, rows: list[dict]) -> tuple[Metrics, list[dict]]:
    pairs = [(r["context"], r["question"]) for r in rows]
    preds = model.predict_many(pairs)
    detail = [{"group_id": r["group_id"], "qtype": r["qtype"], "question": r["question"],
               "gold": r["answer"], "pred": p} for r, p in zip(rows, preds)]
    return qa_metrics(rows, preds), detail


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate a (LoRA) QA model on a split")
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter-dir", default=None, help="omit to score the un-tuned base model")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load_splits()[args.split]
    model = QAModel(base_model=args.base_model, adapter_dir=args.adapter_dir).load()
    metrics, detail = evaluate_split(model, rows)
    print(json.dumps({
        "base_model": args.base_model, "adapter_dir": args.adapter_dir, "split": args.split,
        "exact_match": round(metrics.exact_match, 4), "token_f1": round(metrics.token_f1, 4),
        "per_qtype_em": metrics.per_qtype_em, "per_qtype_f1": metrics.per_qtype_f1,
        "support": metrics.support,
    }, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps({"metrics": metrics.__dict__, "predictions": detail}, indent=2))
        print("Wrote", args.out)
