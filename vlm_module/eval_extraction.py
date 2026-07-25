"""Field-level evaluation of document extraction against ground truth.

Scores an extraction engine (baseline OCR, layout-aware, or Donut VLM) over a
set of form images and reports per-field and overall accuracy / precision /
recall / F1. This is what gives the "OCR enhancement" claim a concrete number:
run the baseline and the enhanced engine over the *same* images and compare.

Matching is normalised (case, whitespace, surrounding punctuation, currency
spacing) so trivially-formatted differences don't count as errors, while a
genuinely wrong value does.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from vlm_module.synth_forms import DATA_DIR, FIELD_KEYS, IMAGES_DIR, load_manifest


def normalise(value: str) -> str:
    v = value.strip().lower()
    v = re.sub(r"\s+", " ", v)
    v = v.strip(" .:;,-_")
    v = v.replace("$ ", "$").replace("s ", "s")  # common OCR spacing slips
    return v


def field_match(pred: str, gold: str) -> bool:
    return normalise(pred) == normalise(gold)


@dataclass
class ExtractionScore:
    engine: str
    n_images: int
    n_fields: int
    exact_field_accuracy: float          # fraction of fields exactly right
    doc_exact_match: float               # fraction of docs with ALL fields right
    precision: float
    recall: float
    f1: float
    per_field_accuracy: dict = field(default_factory=dict)


def evaluate_engine(extract_fn, engine_name: str,
                    images_dir: Path | None = None,
                    manifest: dict | None = None,
                    limit: int | None = None) -> ExtractionScore:
    images_dir = Path(images_dir) if images_dir else IMAGES_DIR
    manifest = manifest if manifest is not None else load_manifest()

    items = list(manifest.items())
    if limit:
        items = items[:limit]

    per_field_correct = {k: 0 for k in FIELD_KEYS}
    total_correct = total_fields = docs_all_correct = 0
    # For P/R/F1 we treat a non-empty predicted field as a "positive".
    tp = fp = fn = 0

    for fname, gold in items:
        img_path = images_dir / fname
        pred = extract_fn(img_path)
        doc_ok = True
        for key in FIELD_KEYS:
            total_fields += 1
            g = gold[key]
            p = pred.get(key, "")
            correct = field_match(p, g)
            if correct:
                per_field_correct[key] += 1
                total_correct += 1
                tp += 1
            else:
                doc_ok = False
                if p.strip():
                    fp += 1  # predicted something, but wrong
                else:
                    fn += 1  # predicted nothing for a field that has a value
        docs_all_correct += int(doc_ok)

    n = len(items)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return ExtractionScore(
        engine=engine_name,
        n_images=n,
        n_fields=total_fields,
        exact_field_accuracy=total_correct / max(total_fields, 1),
        doc_exact_match=docs_all_correct / max(n, 1),
        precision=precision, recall=recall, f1=f1,
        per_field_accuracy={k: per_field_correct[k] / max(n, 1) for k in FIELD_KEYS},
    )


def compare_engines(limit: int | None = None, include_donut: bool = False) -> dict:
    """Baseline OCR vs. layout-aware (vs. Donut if requested). Returns a dict."""
    from vlm_module import baseline_ocr, document_extraction

    results: dict[str, dict] = {}
    manifest = load_manifest()

    if baseline_ocr.tesseract_available():
        s = evaluate_engine(baseline_ocr.extract_fields, "baseline_ocr", manifest=manifest, limit=limit)
        results["baseline_ocr"] = s.__dict__
        s2 = evaluate_engine(lambda p: document_extraction.extract_fields(p, "layout"),
                             "layout_ocr+", manifest=manifest, limit=limit)
        results["layout_ocr+"] = s2.__dict__
    else:
        results["_note"] = "Tesseract binary unavailable — OCR engines skipped."

    if include_donut:
        try:
            s3 = evaluate_engine(lambda p: document_extraction.extract_fields(p, "donut"),
                                 "donut_vlm", manifest=manifest, limit=limit)
            results["donut_vlm"] = s3.__dict__
        except Exception as exc:  # noqa: BLE001
            results["donut_vlm_error"] = str(exc)
    return results


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Compare document-extraction engines on the clean image set")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--donut", action="store_true", help="also run the Donut VLM (heavy download)")
    ap.add_argument("--out", default=str(DATA_DIR / "extraction_scores.json"))
    args = ap.parse_args()

    res = compare_engines(limit=args.limit, include_donut=args.donut)
    Path(args.out).write_text(json.dumps(res, indent=2))
    for name, s in res.items():
        if isinstance(s, dict) and "exact_field_accuracy" in s:
            print(f"{name:16s} field-acc={s['exact_field_accuracy']:.3f}  "
                  f"doc-EM={s['doc_exact_match']:.3f}  F1={s['f1']:.3f}")
        else:
            print(name, s)
    print("Wrote", args.out)
