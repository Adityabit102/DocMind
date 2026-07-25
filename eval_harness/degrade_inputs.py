"""Degradation generators for the noisy / adversarial / real-world harness.

Two families, each parameterised by a severity in [0, 1]:

  TEXT  (feeds the LoRA document classifier)
    - ocr_substitution : swap characters for OCR-confusable ones (o↔0, l↔1, …)
    - drop_words       : randomly delete whole words (missing-word noise)
    - truncate         : cut off the tail of the document (truncated context)
    - keyboard_typos   : adjacent-key substitutions (human typing noise)

  IMAGE (feeds the OCR / VLM extraction module)
    - blur             : Gaussian blur (out-of-focus scan)
    - rotate / skew    : rotation + shear (misaligned scan)
    - downscale        : low-resolution capture (resize down then up)
    - jpeg             : compression artifacts
    - pixel_noise      : additive Gaussian sensor noise

Everything is seeded so a degraded set is reproducible. `apply_text` /
`apply_image` dispatch by name; `TEXT_DEGRADATIONS` / `IMAGE_DEGRADATIONS` list
the available types for the runner to sweep.
"""

from __future__ import annotations

import io
import random

# ── Text degradations ────────────────────────────────────────────────────
_OCR_CONFUSABLE = {
    "o": "0", "O": "0", "l": "1", "I": "1", "i": "1", "s": "5", "S": "5",
    "e": "c", "a": "@", "g": "9", "B": "8", "t": "+", "n": "m",
}
_KEYBOARD_NEIGHBORS = {
    "a": "sq", "b": "vn", "c": "xv", "d": "sf", "e": "wr", "f": "dg",
    "g": "fh", "h": "gj", "i": "uo", "j": "hk", "k": "jl", "l": "k",
    "m": "n", "n": "bm", "o": "ip", "p": "o", "r": "et", "s": "ad",
    "t": "ry", "u": "yi", "v": "cb", "w": "qe", "y": "tu",
}


def ocr_substitution(text: str, severity: float, rng: random.Random) -> str:
    out = []
    for ch in text:
        if ch in _OCR_CONFUSABLE and rng.random() < severity:
            out.append(_OCR_CONFUSABLE[ch])
        else:
            out.append(ch)
    return "".join(out)


def drop_words(text: str, severity: float, rng: random.Random) -> str:
    words = text.split()
    kept = [w for w in words if rng.random() >= severity * 0.6]
    return " ".join(kept) if kept else (words[0] if words else "")


def truncate(text: str, severity: float, rng: random.Random) -> str:
    keep = max(1, int(len(text) * (1.0 - severity * 0.7)))
    return text[:keep]


def keyboard_typos(text: str, severity: float, rng: random.Random) -> str:
    out = []
    for ch in text:
        low = ch.lower()
        if low in _KEYBOARD_NEIGHBORS and rng.random() < severity * 0.5:
            repl = rng.choice(_KEYBOARD_NEIGHBORS[low])
            out.append(repl.upper() if ch.isupper() else repl)
        else:
            out.append(ch)
    return "".join(out)


TEXT_DEGRADATIONS = {
    "ocr_substitution": ocr_substitution,
    "drop_words": drop_words,
    "truncate": truncate,
    "keyboard_typos": keyboard_typos,
}


def apply_text(text: str, kind: str, severity: float, seed: int = 0) -> str:
    if severity <= 0:
        return text
    rng = random.Random(hash((kind, seed, text)) & 0xFFFFFFFF)
    return TEXT_DEGRADATIONS[kind](text, severity, rng)


# ── Image degradations ───────────────────────────────────────────────────
def _blur(img, severity, rng):
    from PIL import ImageFilter

    return img.filter(ImageFilter.GaussianBlur(radius=0.5 + severity * 4.0))


def _rotate(img, severity, rng):
    from PIL import Image

    angle = (rng.random() * 2 - 1) * severity * 12.0  # up to ±12°
    return img.rotate(angle, expand=False, fillcolor="white", resample=Image.BICUBIC)


def _skew(img, severity, rng):
    shear = (rng.random() * 2 - 1) * severity * 0.3
    w, h = img.size
    return img.transform((w, h), 0, (1, shear, 0, 0, 1, 0), fillcolor="white")  # 0 == AFFINE


def _downscale(img, severity, rng):
    w, h = img.size
    factor = 1.0 - severity * 0.75
    small = img.resize((max(1, int(w * factor)), max(1, int(h * factor))))
    return small.resize((w, h))


def _jpeg(img, severity, rng):
    from PIL import Image

    quality = max(3, int(90 - severity * 85))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert(img.mode)


def _pixel_noise(img, severity, rng):
    import numpy as np
    from PIL import Image

    arr = np.asarray(img.convert("L")).astype("float32")
    noise = np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, severity * 70, arr.shape)
    out = np.clip(arr + noise, 0, 255).astype("uint8")
    return Image.fromarray(out).convert(img.mode)


IMAGE_DEGRADATIONS = {
    "blur": _blur,
    "rotate": _rotate,
    "skew": _skew,
    "downscale": _downscale,
    "jpeg": _jpeg,
    "pixel_noise": _pixel_noise,
}


def apply_image(img, kind: str, severity: float, seed: int = 0):
    if severity <= 0:
        return img
    rng = random.Random(hash((kind, seed)) & 0xFFFFFFFF)
    return IMAGE_DEGRADATIONS[kind](img, severity, rng)


if __name__ == "__main__":  # pragma: no cover
    import argparse
    from pathlib import Path

    from PIL import Image

    ap = argparse.ArgumentParser(description="Preview degradations on one image/text")
    ap.add_argument("--image", help="image path to degrade for every image kind")
    ap.add_argument("--severity", type=float, default=0.6)
    ap.add_argument("--out-dir", default="eval_harness/_degrade_preview")
    args = ap.parse_args()

    if args.image:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        base = Image.open(args.image)
        for kind in IMAGE_DEGRADATIONS:
            apply_image(base, kind, args.severity).save(out / f"{kind}.png")
        print("Wrote degraded previews to", out)
    else:
        sample = "INVOICE #4821 From: Acme Logistics  Amount due: $402.11  Net 30."
        for kind in TEXT_DEGRADATIONS:
            print(f"{kind:16s}: {apply_text(sample, kind, args.severity)}")
