"""Run the fine-tuned LLM and the OCR/VLM extractor against clean + degraded
inputs and emit a structured results JSON.

This is the piece that turns "I fine-tuned a model" into "I measured where it
breaks." It produces, for each component, a *clean baseline* plus degradation
curves — accuracy/F1 as a function of severity for every degradation type —
and, for the LLM, two beyond-clean sets (distribution shift, adversarial-lite).

Usage:
  python -m eval_harness.run_evaluation --adapter finetune/adapters/<run> \
         --base-model HuggingFaceTB/SmolLM2-135M
  python -m eval_harness.run_evaluation --skip-vlm          # LLM only
  python -m eval_harness.run_evaluation --skip-llm --vlm-limit 12   # OCR/VLM only

The JSON it writes is consumed by `eval_harness.report` to render the
theme-matched report.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
DEFAULT_SEVERITIES = [0.0, 0.3, 0.6, 0.9]


# ── LLM evaluation ───────────────────────────────────────────────────────
def evaluate_llm(base_model: str, adapter_dir: str,
                 severities: list[float], limit: int | None) -> dict:
    """Extractive-QA evaluation: degrade the CONTEXT (question stays intact),
    score exact-match + token-F1. The primary curve metric is stored under
    ``exact_match`` (mirrored to ``token_f1``) so the report can plot either."""
    from finetune.data_prep import load_splits
    from finetune.eval_task_metrics import QAModel, qa_metrics
    from eval_harness.degrade_inputs import TEXT_DEGRADATIONS, apply_text
    from eval_harness.special_sets import adversarial_lite_set, distribution_shift_set

    rows = load_splits()["test"]
    if limit:
        rows = rows[:limit]

    model = QAModel(base_model=base_model, adapter_dir=adapter_dir).load()

    def score(scored_rows: list[dict]) -> dict:
        preds = model.predict_many([(r["context"], r["question"]) for r in scored_rows])
        m = qa_metrics(scored_rows, preds)
        return {"exact_match": round(m.exact_match, 4), "token_f1": round(m.token_f1, 4)}

    def degrade_context(kind: str, sev: float) -> list[dict]:
        return [{**r, "context": apply_text(r["context"], kind, sev, seed=1)} for r in rows]

    print(f"[LLM] clean baseline on {len(rows)} test QA pairs ...")
    clean = score(rows)
    print(f"      clean EM={clean['exact_match']} token_f1={clean['token_f1']}")

    curves: dict[str, list[dict]] = {}
    for kind in TEXT_DEGRADATIONS:
        pts = []
        for sev in severities:
            if sev == 0.0:
                pts.append({"severity": 0.0, **clean})
                continue
            s = score(degrade_context(kind, sev))
            pts.append({"severity": sev, **s})
            print(f"[LLM] {kind:16s} sev={sev} -> EM={s['exact_match']} f1={s['token_f1']}")
        curves[kind] = pts

    # Beyond-clean sets.
    shift_rows = distribution_shift_set()
    adv_rows = adversarial_lite_set()
    special = {
        "distribution_shift": {**score(shift_rows), "n": len(shift_rows)},
        "adversarial_lite": {**score(adv_rows), "n": len(adv_rows)},
    }
    print(f"[LLM] distribution_shift EM={special['distribution_shift']['exact_match']} | "
          f"adversarial_lite EM={special['adversarial_lite']['exact_match']}")

    return {
        "base_model": base_model,
        "adapter_dir": adapter_dir,
        "task": "document extractive QA",
        "n_test": len(rows),
        "severities": severities,
        "clean": clean,
        "degradation_curves": curves,
        "special_sets": special,
    }


# ── VLM / OCR evaluation ─────────────────────────────────────────────────
def evaluate_vlm(severities: list[float], limit: int | None, include_donut: bool) -> dict:
    from PIL import Image

    from eval_harness.degrade_inputs import IMAGE_DEGRADATIONS, apply_image
    from vlm_module import baseline_ocr, document_extraction
    from vlm_module.eval_extraction import FIELD_KEYS, field_match, load_manifest
    from vlm_module.synth_forms import IMAGES_DIR

    if not baseline_ocr.tesseract_available():
        return {"skipped": "Tesseract binary not available — OCR/VLM eval skipped."}

    manifest = load_manifest()
    items = list(manifest.items())
    if limit:
        items = items[:limit]

    engines = {
        "baseline_ocr": lambda img: baseline_ocr._parse_pairs(
            __import__("pytesseract").image_to_string(img)),
        "layout_ocr+": lambda img: _extract_from_pil(document_extraction, img, "layout"),
    }
    if include_donut:
        engines["donut_vlm"] = lambda img: _extract_from_pil(document_extraction, img, "donut")

    def field_f1_over(images_and_gold, engine_fn) -> dict:
        tp = fp = fn = correct = total = 0
        for pil, gold in images_and_gold:
            pred = engine_fn(pil)
            for key in FIELD_KEYS:
                total += 1
                if field_match(pred.get(key, ""), gold[key]):
                    correct += 1
                    tp += 1
                elif pred.get(key, "").strip():
                    fp += 1
                else:
                    fn += 1
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return {"field_accuracy": round(correct / max(total, 1), 4), "f1": round(f1, 4)}

    base_imgs = [(Image.open(IMAGES_DIR / f).convert("RGB"), g) for f, g in items]

    print(f"[VLM] clean baseline on {len(items)} images, engines={list(engines)}")
    clean = {}
    for name, fn in engines.items():
        clean[name] = field_f1_over(base_imgs, fn)
        print(f"      {name:14s} field_acc={clean[name]['field_accuracy']} f1={clean[name]['f1']}")

    curves: dict[str, dict[str, list]] = {}
    for kind in IMAGE_DEGRADATIONS:
        curves[kind] = {name: [] for name in engines}
        for sev in severities:
            if sev == 0.0:
                imgs = base_imgs
            else:
                imgs = [(apply_image(img, kind, sev, seed=3), g) for img, g in base_imgs]
            for name, fn in engines.items():
                pt = {"severity": sev, **field_f1_over(imgs, fn)}
                curves[kind][name].append(pt)
            row = " ".join(f"{n}={curves[kind][n][-1]['field_accuracy']}" for n in engines)
            print(f"[VLM] {kind:10s} sev={sev} -> {row}")

    return {
        "engines": list(engines),
        "n_images": len(items),
        "severities": severities,
        "clean": clean,
        "degradation_curves": curves,
    }


def _extract_from_pil(document_extraction, pil_img, engine):
    """The extraction fns take a path; write the (possibly-degraded) PIL to a temp file."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        pil_img.convert("RGB").save(tf.name)
        path = tf.name
    try:
        return document_extraction.extract_fields(path, engine=engine)
    finally:
        Path(path).unlink(missing_ok=True)


# ── Orchestration ────────────────────────────────────────────────────────
def _latest_adapter() -> str | None:
    adapters = Path("finetune/adapters")
    runs = [p for p in adapters.glob("*") if (p / "adapter_config.json").exists()]
    if not runs:
        return None
    return str(max(runs, key=lambda p: p.stat().st_mtime))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run clean+degraded evaluation of the LLM and VLM modules")
    ap.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (default: latest under finetune/adapters)")
    ap.add_argument("--severities", type=float, nargs="+", default=DEFAULT_SEVERITIES)
    ap.add_argument("--llm-limit", type=int, default=None, help="cap test docs for speed")
    ap.add_argument("--vlm-limit", type=int, default=12, help="cap images for speed")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--skip-vlm", action="store_true")
    ap.add_argument("--donut", action="store_true", help="also evaluate the Donut VLM engine")
    ap.add_argument("--out", default=str(REPORTS_DIR / "eval_results.json"))
    args = ap.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "severities": args.severities}

    if not args.skip_llm:
        adapter = args.adapter or _latest_adapter()
        if not adapter:
            print("No adapter found — run finetune.train_lora first, or pass --adapter.")
        else:
            results["llm"] = evaluate_llm(args.base_model, adapter, args.severities, args.llm_limit)

    if not args.skip_vlm:
        results["vlm"] = evaluate_vlm(args.severities, args.vlm_limit, args.donut)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print("\nWrote results ->", args.out)


if __name__ == "__main__":
    main()
