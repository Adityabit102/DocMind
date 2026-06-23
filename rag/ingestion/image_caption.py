"""Vision captioning for images embedded in PDFs.

When ``enable_image_captioning`` is set, each PDF page is rasterised and passed
to a vision-capable model (OpenAI ``gpt-4o`` family, or a local Ollama vision
model such as ``llava``) which describes any figures, charts, or diagrams. The
captions are appended to the corpus as extra ``Document`` chunks so retrieval
can surface visual content that has no text layer.

Best-effort and gracefully degrading: disabled by default, a no-op when the
provider has no vision model or the rasteriser is unavailable, so the local-first
zero-key path is never blocked.
"""

from __future__ import annotations

import base64
import io
import logging

from langchain_core.documents import Document

from app.config import Settings, get_settings

logger = logging.getLogger("docmind")

_CAPTION_PROMPT = (
    "Describe any figures, charts, tables, or diagrams in this image in 2-4 "
    "sentences. If it is plain text only, reply with 'NO_VISUAL_CONTENT'."
)


def _encode_png(image) -> str:  # type: ignore[no-untyped-def]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _vision_llm(settings: Settings):  # type: ignore[no-untyped-def]
    """Return a vision-capable chat model, or None if none is configured."""
    if settings.llm_provider == "openai" and settings.has_openai:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model="gpt-4o", api_key=settings.openai_api_key, temperature=0)
    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=settings.vision_model, base_url=settings.ollama_base_url)
    return None


def caption_pdf_images(path: str, settings: Settings | None = None) -> list[Document]:
    """Caption visual content per PDF page; return one caption ``Document`` each.

    Returns an empty list (no-op) when captioning is disabled, no vision model is
    available, or rasterisation/inference fails.
    """
    settings = settings or get_settings()
    if not settings.enable_image_captioning:
        return []

    llm = _vision_llm(settings)
    if llm is None:
        logger.info("Image captioning enabled but no vision model is configured — skipping")
        return []

    try:
        import os

        from pdf2image import convert_from_path

        pages = convert_from_path(path)
    except Exception as exc:  # noqa: BLE001 — rasteriser/poppler missing
        logger.warning("Image captioning skipped (rasteriser unavailable): %s", exc)
        return []

    filename = os.path.basename(path)
    captions: list[Document] = []
    for idx, image in enumerate(pages):
        try:
            b64 = _encode_png(image)
            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": _CAPTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
            result = llm.invoke([message])
            text = getattr(result, "content", str(result)).strip()
        except Exception as exc:  # noqa: BLE001 — per-page best-effort
            logger.warning("Caption failed for %s page %d: %s", filename, idx + 1, exc)
            continue
        if not text or "NO_VISUAL_CONTENT" in text:
            continue
        captions.append(
            Document(
                page_content=f"[Image description] {text}",
                metadata={
                    "source": filename,
                    "filename": filename,
                    "page_number": idx + 1,
                    "image_caption": True,
                },
            )
        )
    return captions
