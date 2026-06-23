FROM python:3.11-slim

WORKDIR /app

# System dependencies for OCR (pytesseract) and PDF rasterisation (pdf2image),
# plus libgl for unstructured/Pillow image handling.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        poppler-utils \
        libgl1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install the CPU-only torch wheel first so the heavy CUDA/nvidia packages are
# never pulled (inference here runs on CPU). The pinned torch then satisfies the
# requirement below without re-resolving to the multi-GB CUDA build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
