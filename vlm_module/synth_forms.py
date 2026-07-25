"""Synthesise document-image forms with exact key/value ground truth.

Document-image datasets with clean field-level ground truth (FUNSD, CORD, …)
are the right target, but they need network + licence acceptance and their GT
is noisy. To keep the VLM/OCR pipeline offline-reproducible AND give the
degradation harness a set of images with *known* answers, we render our own
form images from seeded data. Each image ships with the exact field dict it
was drawn from, so field-level accuracy is unambiguous.

The same rendered images are the inputs the eval harness later degrades
(blur / skew / low-res / noise) to measure OCR & VLM robustness.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
IMAGES_DIR = DATA_DIR / "images"

_FIRST = ["Ava", "Liam", "Priya", "Kenji", "Sofia", "Omar", "Elena", "Marcus",
          "Aisha", "Diego", "Hana", "Tariq", "Lena", "Ravi", "Noah", "Mia"]
_LAST = ["Osei", "Nakamura", "Delgado", "Kowalski", "Abebe", "Rossi", "Haddad",
         "Fischer", "Okonkwo", "Petrov", "Silva", "Ahmed", "Novak", "Reyes"]
_VENDOR = ["Northwind Traders", "Acme Logistics", "Blue Harbor Foods",
           "Meridian Analytics", "Orbital Systems", "Sunfield Grocers"]
_CITY = ["Portland", "Leeds", "Nagoya", "Turin", "Accra", "Valencia", "Austin"]

# Field keys are fixed so scoring is a clean per-field comparison.
FIELD_KEYS = ["Invoice No", "Date", "Vendor", "Bill To", "City", "Total"]


def _make_fields(rng: random.Random) -> dict[str, str]:
    return {
        "Invoice No": f"INV-{rng.randint(10000, 99999)}",
        "Date": f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        "Vendor": rng.choice(_VENDOR),
        "Bill To": f"{rng.choice(_FIRST)} {rng.choice(_LAST)}",
        "City": rng.choice(_CITY),
        "Total": f"${rng.randint(50, 9999)}.{rng.randint(0, 99):02d}",
    }


def _load_font(size: int):
    from PIL import ImageFont

    # Prefer a real TTF for crisp glyphs OCR can read; fall back to bitmap.
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()


def render_form(fields: dict[str, str], path: Path) -> None:
    from PIL import Image, ImageDraw

    W, H = 800, 520
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    title_font = _load_font(30)
    key_font = _load_font(22)
    val_font = _load_font(22)

    d.text((40, 30), "INVOICE", font=title_font, fill="black")
    d.line((40, 80, W - 40, 80), fill="black", width=2)

    # Draw "Key:  Value" — key and value on the SAME text line with a modest,
    # measured gap. A large empty column gap makes Tesseract segment keys and
    # values into separate blocks (a real form-OCR failure mode), so we keep the
    # value close enough to stay on one line while still being a distinct token.
    y = 120
    for key in FIELD_KEYS:
        label = f"{key}:"
        d.text((60, y), label, font=key_font, fill="black")
        try:
            key_w = int(d.textlength(label, font=key_font))
        except Exception:
            key_w = 180
        d.text((60 + key_w + 28, y), fields[key], font=val_font, fill="black")
        y += 58

    d.line((40, H - 50, W - 40, H - 50), fill="black", width=1)
    d.text((60, H - 40), "Thank you for your business.", font=_load_font(16), fill="black")
    img.save(path)


def build_image_dataset(n: int = 24, seed: int = 7, out_dir: Path | None = None) -> Path:
    """Render ``n`` form images + a ground_truth.json manifest. Returns manifest path."""
    out_dir = Path(out_dir) if out_dir else DATA_DIR
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    manifest = {}
    for i in range(n):
        fields = _make_fields(rng)
        fname = f"form_{i:03d}.png"
        render_form(fields, images_dir / fname)
        manifest[fname] = fields

    manifest_path = out_dir / "ground_truth.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def load_manifest(out_dir: Path | None = None) -> dict[str, dict]:
    out_dir = Path(out_dir) if out_dir else DATA_DIR
    manifest_path = out_dir / "ground_truth.json"
    if not manifest_path.exists():
        build_image_dataset(out_dir=out_dir)
    return json.loads(manifest_path.read_text())


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Render synthetic form images + ground truth")
    ap.add_argument("-n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    mp = build_image_dataset(n=args.n, seed=args.seed)
    print(f"Rendered {args.n} forms -> {IMAGES_DIR}")
    print("Ground truth manifest ->", mp)
