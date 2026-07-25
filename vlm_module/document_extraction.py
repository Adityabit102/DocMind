"""Document-understanding extraction — the "enhanced" side of the comparison.

Two engines, selectable with ``--engine``:

  layout  (default, always runnable)
      A lightweight document-understanding pipeline: image preprocessing
      (grayscale → upscale → adaptive binarisation) to make text OCR-legible
      even when degraded, then *layout-aware* field extraction using
      Tesseract's word-level bounding boxes (``image_to_data``). Instead of
      regex-ing flat text, we reconstruct lines from box geometry and split
      each key's value by the horizontal gap after the key — far more robust to
      skew, multi-column drift, and noise than the flat-text baseline.

  donut   (optional, heavy)
      A genuine Vision-Language Model: naver-clova-ix Donut (an OCR-free
      image→text document parser). This is the real "VLM" engine. It downloads
      ~800 MB and is slow on CPU, so it is opt-in. Whether the reported numbers
      come from `layout` or `donut` is stated honestly in the writeup — the
      design doc explicitly requires not blurring that choice (§5.2).

Both return a ``{field: value}`` dict scored by ``eval_extraction``.
"""

from __future__ import annotations

import re
from pathlib import Path

from vlm_module.synth_forms import FIELD_KEYS

# ── Shared image preprocessing ───────────────────────────────────────────
def preprocess(image_path: str | Path):
    """Gently condition a (possibly degraded) scan to help OCR — without
    destroying it.

    Deliberately *non-destructive*: we upscale low-resolution scans so glyphs
    have enough pixels, stretch contrast, and knock down speckle with a light
    median filter — then hand a *grayscale* image to Tesseract and let its own
    (Otsu) binariser decide the threshold. An earlier version hard-thresholded
    here and it wiped out blurred/low-contrast text, hurting more than it helped;
    letting Tesseract binarise is both simpler and more robust.
    """
    from PIL import Image, ImageFilter, ImageOps

    img = Image.open(image_path).convert("L")
    # Upscale small/low-res scans so glyphs have enough pixels for the engine.
    if max(img.size) < 1400:
        scale = 1400 / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.MedianFilter(size=3))  # denoise speckle/sensor noise
    return img


# ── Engine: layout-aware OCR ─────────────────────────────────────────────
def _ocr_words(pil_img):
    """Return word records: (text, left, top, right, bottom, line_key)."""
    import pytesseract
    from pytesseract import Output

    data = pytesseract.image_to_data(pil_img, output_type=Output.DICT)
    words = []
    for i, txt in enumerate(data["text"]):
        if not txt.strip():
            continue
        if int(data.get("conf", ["-1"])[i] if isinstance(data["conf"][i], str) else data["conf"][i]) < 0:
            continue
        left, top, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        line_key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        words.append({"text": txt, "left": left, "top": top,
                      "right": left + w, "bottom": top + h, "line": line_key})
    return words


def _group_lines(words):
    """Cluster words into visual rows by their vertical centre.

    We deliberately do NOT trust Tesseract's own block/line grouping: on
    two-column forms it splits the key column and value column into separate
    blocks, orphaning every value from its key. Instead we bin words by their
    y-centre into rows, so a key and its value on the same physical line are
    reunited regardless of how far apart they sit horizontally.
    """
    if not words:
        return []
    ws = sorted(words, key=lambda w: (w["top"] + w["bottom"]) / 2)
    heights = sorted((w["bottom"] - w["top"]) for w in ws)
    med_h = heights[len(heights) // 2] or 12
    tol = med_h * 0.6  # words within ~0.6 of a line-height share a row

    rows: list[list[dict]] = []
    current: list[dict] = [ws[0]]
    current_c = (ws[0]["top"] + ws[0]["bottom"]) / 2
    for w in ws[1:]:
        c = (w["top"] + w["bottom"]) / 2
        if abs(c - current_c) <= tol:
            current.append(w)
        else:
            rows.append(sorted(current, key=lambda x: x["left"]))
            current = [w]
        current_c = c
    rows.append(sorted(current, key=lambda x: x["left"]))
    return rows


def extract_fields_layout(image_path: str | Path) -> dict[str, str]:
    """Layout-aware extraction: find the key words on a line, take words to the right."""
    img = preprocess(image_path)
    lines = _group_lines(_ocr_words(img))
    fields = {k: "" for k in FIELD_KEYS}

    for key in FIELD_KEYS:
        key_tokens = key.lower().split()
        for ws in lines:
            texts = [w["text"].strip().strip(":;").lower() for w in ws]
            # Find a contiguous run of words matching the key tokens.
            idx = _find_subseq(texts, key_tokens)
            if idx is None:
                continue
            key_end = idx + len(key_tokens)
            value_words = [w["text"] for w in ws[key_end:]]
            value = " ".join(value_words).strip().lstrip(":;").strip()
            if value:
                fields[key] = value
                break
    return fields


def _find_subseq(hay: list[str], needle: list[str]):
    for i in range(len(hay) - len(needle) + 1):
        if all(needle[j] in hay[i + j] for j in range(len(needle))):
            return i
    return None


# ── Engine: Donut VLM (optional) ─────────────────────────────────────────
_DONUT_CACHE: dict = {}

# If the weights have been fetched to a local dir (see MODEL_BUILDING.md — the
# HF hub downloader stalls on the 806MB blob over anonymous connections, so we
# stream it manually), load from there. Otherwise fall back to the hub id.
# This only changes *where the model is loaded from*, never the extraction logic.
_LOCAL_DONUT = Path(__file__).resolve().parent.parent / "models" / "donut-cord"
_HUB_DONUT = "naver-clova-ix/donut-base-finetuned-cord-v2"


def _default_donut_model() -> str:
    if (_LOCAL_DONUT / "pytorch_model.bin").exists() or (_LOCAL_DONUT / "model.safetensors").exists():
        return str(_LOCAL_DONUT)
    return _HUB_DONUT


def _load_donut(model_name: str):
    if model_name in _DONUT_CACHE:
        return _DONUT_CACHE[model_name]
    import torch
    from transformers import DonutProcessor, VisionEncoderDecoderModel

    processor = DonutProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name).eval()
    _DONUT_CACHE[model_name] = (processor, model, torch)
    return _DONUT_CACHE[model_name]


def extract_fields_donut(image_path: str | Path,
                         model_name: str | None = None) -> dict[str, str]:
    """Run the Donut VLM and map its parsed output onto our field schema."""
    from PIL import Image

    processor, model, torch = _load_donut(model_name or _default_donut_model())
    image = Image.open(image_path).convert("RGB")
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
    pixel_values = processor(image, return_tensors="pt").pixel_values
    with torch.no_grad():
        out = model.generate(
            pixel_values, decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_position_embeddings,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True, num_beams=1,
        )
    seq = processor.batch_decode(out)[0]
    seq = re.sub(r"<.*?>", " ", seq)  # strip Donut's XML-ish tags
    # Donut-CORD emits free receipt text; map any of our known values that appear.
    return _fuzzy_map_to_fields(seq)


def _fuzzy_map_to_fields(text: str) -> dict[str, str]:
    fields = {k: "" for k in FIELD_KEYS}
    for key in FIELD_KEYS:
        m = re.search(rf"{re.escape(key)}\s*[:;]?\s*([^\n]+)", text, re.IGNORECASE)
        if m:
            fields[key] = m.group(1).strip()
    return fields


# ── Dispatch ─────────────────────────────────────────────────────────────
def extract_fields(image_path: str | Path, engine: str = "layout") -> dict[str, str]:
    if engine == "layout":
        return extract_fields_layout(image_path)
    if engine == "donut":
        return extract_fields_donut(image_path)
    raise ValueError(f"unknown engine {engine!r} (use 'layout' or 'donut')")


if __name__ == "__main__":  # pragma: no cover
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Enhanced document field extraction")
    ap.add_argument("image")
    ap.add_argument("--engine", default="layout", choices=["layout", "donut"])
    args = ap.parse_args()
    print(json.dumps(extract_fields(args.image, engine=args.engine), indent=2))
