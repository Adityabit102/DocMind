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


# Three GENUINELY distinct visual layouts (not just field-value variation) —
# different page geometry, font family, header style, and — critically for
# "stacked" — key/value on DIFFERENT lines rather than the same line. All three
# still carry the identical FIELD_KEYS schema, so every extraction engine is
# scored identically regardless of which template an image came from.
TEMPLATES = ["twocol", "table", "stacked"]


def _load_font(size: int, family: str = "sans"):
    from PIL import ImageFont

    candidates = {
        "sans": ("/System/Library/Fonts/Supplemental/Arial.ttf",
                "/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"),
        "serif": ("/System/Library/Fonts/Supplemental/Georgia.ttf",
                 "/System/Library/Fonts/Supplemental/Georgia Bold.ttf"),
        "mono": ("/System/Library/Fonts/Supplemental/Courier New.ttf",
                "/System/Library/Fonts/Supplemental/Courier New Bold.ttf"),
    }
    for candidate in candidates.get(family, candidates["sans"]):
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return _load_font(size, "sans") if family != "sans" else ImageFont.load_default()


def _render_twocol(fields: dict[str, str], path: Path) -> None:
    """Original layout: sans-serif, key: value on ONE line, header top-left."""
    from PIL import Image, ImageDraw

    W, H = 800, 520
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    title_font = _load_font(30, "sans")
    key_font = _load_font(22, "sans")
    val_font = _load_font(22, "sans")

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
    d.text((60, H - 40), "Thank you for your business.", font=_load_font(16, "sans"), fill="black")
    img.save(path)


def _render_table(fields: dict[str, str], path: Path) -> None:
    """Bordered table layout: serif font, centered banner header, ruled rows,
    Field/Value COLUMNS with a vertical divider — visually distinct geometry
    and typography from twocol, wider canvas."""
    from PIL import Image, ImageDraw

    W, H = 900, 600
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    title_font = _load_font(28, "serif")
    header_font = _load_font(18, "serif")
    key_font = _load_font(20, "serif")
    val_font = _load_font(20, "serif")

    title = "STATEMENT"
    tw = d.textlength(title, font=title_font)
    d.text(((W - tw) / 2, 24), title, font=title_font, fill="black")
    d.rectangle((50, 70, W - 50, 74), fill="black")

    table_top, row_h = 100, 46
    col_split = 300
    d.rectangle((50, table_top, W - 50, table_top + row_h * (len(FIELD_KEYS) + 1)), outline="black", width=2)
    d.line((col_split, table_top, col_split, table_top + row_h * (len(FIELD_KEYS) + 1)), fill="black", width=2)
    d.text((60, table_top + 12), "Field", font=header_font, fill="black")
    d.text((col_split + 20, table_top + 12), "Value", font=header_font, fill="black")
    d.line((50, table_top + row_h, W - 50, table_top + row_h), fill="black", width=2)

    for i, key in enumerate(FIELD_KEYS):
        row_y = table_top + row_h * (i + 1)
        d.text((60, row_y + 12), key, font=key_font, fill="black")
        d.text((col_split + 20, row_y + 12), fields[key], font=val_font, fill="black")
        if i > 0:
            d.line((50, row_y, W - 50, row_y), fill="#888888", width=1)

    d.text((50, H - 40), "This statement summarises your recent account activity.",
           font=_load_font(15, "serif"), fill="black")
    img.save(path)


def _render_stacked(fields: dict[str, str], path: Path) -> None:
    """Narrow, receipt-style layout: monospace font, key and value on
    DIFFERENT lines (value indented below its label), dashed separators,
    centered header — the most different geometry of the three, deliberately,
    so a model trained without seeing it faces a genuine layout shift."""
    from PIL import Image, ImageDraw

    W, H = 520, 760
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    title_font = _load_font(24, "mono")
    key_font = _load_font(18, "mono")
    val_font = _load_font(20, "mono")

    title = "DOCUMENT SUMMARY"
    tw = d.textlength(title, font=title_font)
    d.text(((W - tw) / 2, 30), title, font=title_font, fill="black")
    dash = "-" * 46
    dw = d.textlength(dash, font=key_font)
    d.text(((W - dw) / 2, 74), dash, font=key_font, fill="black")

    y = 110
    for key in FIELD_KEYS:
        d.text((40, y), f"{key.upper()}:", font=key_font, fill="black")
        y += 28
        d.text((60, y), fields[key], font=val_font, fill="black")
        y += 34
        d.text(((W - dw) / 2, y), dash, font=key_font, fill="#666666")
        y += 30

    d.text((40, H - 50), "Retain for your records.", font=_load_font(14, "mono"), fill="black")
    img.save(path)


_TEMPLATE_RENDERERS = {"twocol": _render_twocol, "table": _render_table, "stacked": _render_stacked}


def render_form(fields: dict[str, str], path: Path, template: str = "twocol") -> None:
    if template not in _TEMPLATE_RENDERERS:
        raise ValueError(f"unknown template {template!r} (use one of {TEMPLATES})")
    _TEMPLATE_RENDERERS[template](fields, path)


def build_image_dataset(n: int = 24, seed: int = 7, out_dir: Path | None = None,
                        templates: list[str] | None = None) -> Path:
    """Render ``n`` form images (cycling through ``templates``) + a
    ground_truth.json manifest (each entry also carries a ``_template`` tag).
    Returns the manifest path. ``templates`` defaults to ``["twocol"]`` —
    the original single-layout behaviour — for backward compatibility."""
    out_dir = Path(out_dir) if out_dir else DATA_DIR
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    templates = templates or ["twocol"]

    manifest = {}
    for i in range(n):
        fields = _make_fields(rng)
        template = templates[i % len(templates)]
        fname = f"form_{i:03d}.png"
        render_form(fields, images_dir / fname, template=template)
        manifest[fname] = {**fields, "_template": template}

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
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--templates", nargs="+", default=None, choices=TEMPLATES,
                    help=f"cycle through these templates (default: twocol only); choices: {TEMPLATES}")
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else DATA_DIR
    mp = build_image_dataset(n=args.n, seed=args.seed, out_dir=out_dir, templates=args.templates)
    print(f"Rendered {args.n} forms (templates={args.templates or ['twocol']}) -> {out_dir / 'images'}")
    print("Ground truth manifest ->", mp)
