"""Diagnose *why* the zero-shot Donut VLM scores 0.00 on this field schema.

Donut here is run ZERO-SHOT (pretrained `donut-base-finetuned-cord-v2`, not
fine-tuned by us). It scores 0.00 schema-aligned field accuracy — this script
shows that is an output-format / schema mismatch, NOT an OCR failure or a broken
eval pipeline, by measuring two things over the same images:

  raw_value_recall            — fraction of gold field VALUES that appear
                                *anywhere* in Donut's decoded sequence (i.e. did
                                Donut's OCR read the value at all, ignoring which
                                field name it was tagged under).
  schema_aligned_field_accuracy — fraction mapped to the CORRECT field name
                                (what eval_extraction actually scores).

A high raw recall with ~zero schema accuracy is the signature of a schema
mismatch: Donut-CORD emits receipt tags (<s_menu>, <s_total_price>, …) instead
of this form's keys, so the content is present but mis-labelled.

Run:  python -m vlm_module.donut_diagnostic
Writes: vlm_module/data/donut_diagnostic.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from vlm_module import document_extraction as de
from vlm_module.synth_forms import DATA_DIR, FIELD_KEYS, IMAGES_DIR, load_manifest


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def run(limit: int = 12) -> dict:
    from PIL import Image

    manifest = load_manifest()
    items = list(manifest.items())[:limit]
    proc, model, torch = de._load_donut(de._default_donut_model())

    raw_hits = schema_hits = total = 0
    examples = []
    for fname, gold in items:
        img = Image.open(IMAGES_DIR / fname).convert("RGB")
        dii = proc.tokenizer("<s_cord-v2>", add_special_tokens=False, return_tensors="pt").input_ids
        pv = proc(img, return_tensors="pt").pixel_values
        with torch.no_grad():
            out = model.generate(
                pv, decoder_input_ids=dii,
                max_length=model.decoder.config.max_position_embeddings,
                pad_token_id=proc.tokenizer.pad_token_id,
                eos_token_id=proc.tokenizer.eos_token_id, use_cache=True, num_beams=1)
        seq_norm = _norm(proc.batch_decode(out)[0])
        mapped = de.extract_fields(IMAGES_DIR / fname, "donut")
        for k in FIELD_KEYS:
            total += 1
            if _norm(gold[k]) and _norm(gold[k]) in seq_norm:
                raw_hits += 1
            if _norm(mapped.get(k, "")) == _norm(gold[k]):
                schema_hits += 1
        if len(examples) < 1:
            examples.append({"file": fname, "gold": gold, "mapped_to_our_schema": mapped})

    return {
        "engine": "donut-base-finetuned-cord-v2 (zero-shot, not fine-tuned)",
        "images": len(items),
        "total_fields": total,
        "raw_value_recall": round(raw_hits / total, 3),
        "schema_aligned_field_accuracy": round(schema_hits / total, 3),
        "raw_hits": raw_hits,
        "schema_hits": schema_hits,
        "interpretation": ("Donut's OCR reads a majority of values but tags them "
                           "under the CORD receipt schema, so they never align to "
                           "this form's field names -> schema mismatch, not OCR failure."),
        "example": examples[0] if examples else None,
    }


if __name__ == "__main__":  # pragma: no cover
    res = run()
    out = Path(DATA_DIR) / "donut_diagnostic.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "example"}, indent=2))
    print("Wrote", out)
