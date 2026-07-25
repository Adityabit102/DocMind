"""Plain-OCR baseline for document field extraction.

This is the deliberately-simple point of comparison the "enhancement" claim is
measured against (design doc §5.2): run Tesseract on the raw image, then pull
``Key: Value`` pairs out of the flat text with a regex. No preprocessing, no
layout awareness. If the Tesseract binary is missing the module degrades
gracefully — `tesseract_available()` returns False and callers skip it rather
than crashing the harness.
"""

from __future__ import annotations

import re
from pathlib import Path

from vlm_module.synth_forms import FIELD_KEYS


def tesseract_available() -> bool:
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def ocr_text(image_path: str | Path) -> str:
    import pytesseract
    from PIL import Image

    return pytesseract.image_to_string(Image.open(image_path))


def _parse_pairs(text: str) -> dict[str, str]:
    """Extract known ``Key: Value`` fields from flat OCR text.

    We look for each expected key (case-insensitively) and grab the remainder
    of that line as the value. This is intentionally naive — it is the baseline.
    """
    fields: dict[str, str] = {k: "" for k in FIELD_KEYS}
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for key in FIELD_KEYS:
        pat = re.compile(rf"{re.escape(key)}\s*[:;]?\s*(.+)", re.IGNORECASE)
        for ln in lines:
            m = pat.search(ln)
            if m:
                fields[key] = m.group(1).strip()
                break
    return fields


def extract_fields(image_path: str | Path) -> dict[str, str]:
    """Baseline extraction: raw OCR -> regex key/value parse."""
    if not tesseract_available():
        raise RuntimeError("Tesseract binary not found; install it to run the OCR baseline.")
    return _parse_pairs(ocr_text(image_path))


if __name__ == "__main__":  # pragma: no cover
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Baseline OCR field extraction on one image")
    ap.add_argument("image")
    args = ap.parse_args()
    print("tesseract available:", tesseract_available())
    print(json.dumps(extract_fields(args.image), indent=2))
