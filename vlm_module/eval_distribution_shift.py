"""Evaluate all engines on a document LAYOUT never seen during Donut's
LoRA fine-tune — the VLM-side analogue of the LLM harness's
``distribution_shift_set`` (eval_harness/special_sets.py).

Training uses the "twocol" and "table" templates (vlm_module/data_donut_train/).
This script scores every engine on "stacked" (vlm_module/data_ood_shift/) — a
visually distinct layout (monospace, key/value on separate lines) that the
LoRA adapter never saw. A real, unrigged distribution shift: even the
regex/OCR baselines are NOT guaranteed to handle it (they don't — see
TECHNICAL_REPORT.md §6.2).

Run: python -m vlm_module.eval_distribution_shift --donut --donut-finetuned
"""

from __future__ import annotations

import json
from pathlib import Path

OOD_DIR = Path(__file__).resolve().parent / "data_ood_shift"


def run(include_donut: bool = False, include_donut_finetuned: bool = False) -> dict:
    from vlm_module import baseline_ocr, document_extraction
    from vlm_module.eval_extraction import evaluate_engine
    from vlm_module.synth_forms import load_manifest

    manifest = load_manifest(out_dir=OOD_DIR)
    images_dir = OOD_DIR / "images"
    results: dict[str, dict] = {}

    s = evaluate_engine(baseline_ocr.extract_fields, "baseline_ocr", images_dir=images_dir, manifest=manifest)
    results["baseline_ocr"] = s.__dict__
    s2 = evaluate_engine(lambda p: document_extraction.extract_fields(p, "layout"),
                         "layout_ocr+", images_dir=images_dir, manifest=manifest)
    results["layout_ocr+"] = s2.__dict__

    if include_donut:
        s3 = evaluate_engine(lambda p: document_extraction.extract_fields(p, "donut"),
                             "donut_vlm", images_dir=images_dir, manifest=manifest)
        results["donut_vlm"] = s3.__dict__
    if include_donut_finetuned:
        s4 = evaluate_engine(lambda p: document_extraction.extract_fields(p, "donut_finetuned"),
                             "donut_finetuned", images_dir=images_dir, manifest=manifest)
        results["donut_finetuned"] = s4.__dict__

    return results


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate engines on the unseen-template distribution-shift set")
    ap.add_argument("--donut", action="store_true")
    ap.add_argument("--donut-finetuned", action="store_true")
    ap.add_argument("--out", default=str(OOD_DIR / "distribution_shift_scores.json"))
    args = ap.parse_args()

    res = run(include_donut=args.donut, include_donut_finetuned=args.donut_finetuned)
    Path(args.out).write_text(json.dumps(res, indent=2))
    for name, s in res.items():
        print(f"{name:16s} field-acc={s['exact_field_accuracy']:.3f}  "
              f"doc-EM={s['doc_exact_match']:.3f}  F1={s['f1']:.3f}")
    print("Wrote", args.out)
