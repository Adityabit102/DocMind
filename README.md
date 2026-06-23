---
title: DocMind
emoji: 📄
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
---

<!-- The YAML block above configures Hugging Face Spaces (Docker SDK). It is
     ignored by normal Markdown viewers and safe to keep in the repo. -->

# DocMind — AI-Powered RAG Document Q&A System

DocMind ingests your documents (PDF, DOCX, TXT, MD, CSV, HTML) and answers
natural-language questions with **grounded, cited answers**. Every claim is
traceable to a source page and chunk. It implements production RAG end-to-end:
hybrid retrieval (dense FAISS + sparse BM25 + RRF), cross-encoder reranking,
HyDE, query expansion, CRAG/Self-RAG, semantic caching, RAGAS evaluation,
drift detection, and full observability.

- **Backend:** FastAPI + LangChain (LCEL) + FAISS + BM25 — REST API with SSE streaming.
- **Frontend:** Next.js + React (Three.js / React Three Fiber 3D, Framer Motion), custom-crafted.
- **Local-first:** runs with **no API keys** (HuggingFace embeddings + Ollama). OpenAI/Anthropic optional.

---

## Quick start (local, no keys)

```bash
# 1. Backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # Python 3.11 (matches Docker)
# On Python 3.12 / 3.13 use the loosened, wheel-available pins instead:
#   pip install -r requirements-py313.txt
cp .env.example .env                      # defaults are local-first
uvicorn app.main:app --reload --port 8000
#   → API docs at http://localhost:8000/docs

# 2. local LLM for answers WITHOUT an API key (pick one)
#   a) Ollama — install from https://ollama.com (or `brew install ollama`)
ollama serve &        # start the daemon
ollama pull llama3    # default model used by .env
#   b) OR set OPENAI_API_KEY / ANTHROPIC_API_KEY in .env and
#      change LLM_PROVIDER accordingly — no Ollama needed.
# The UI runs and browses fine even before this; only live answers need an LLM.

# 3. Frontend
cd frontend && npm install && npm run dev
#   → UI at http://localhost:3000
```

> **Python version note.** `requirements.txt` pins the spec versions for **Python 3.11**.
> Some of those pins (e.g. `faiss-cpu==1.8.0`) have no wheels for 3.12/3.13 — use
> `requirements-py313.txt` there. Docker always uses 3.11 and the pinned file.

### Docker (full stack)

```bash
docker compose up --build -d                      # backend + frontend + redis
docker compose --profile local-llm up --build -d  # also start Ollama
```

---

## Design system

Custom-crafted, **no glassmorphism / no gradients**. Warm taupe → cream palette:

| Token | Hex | Use |
|-------|-----|-----|
| `--ink` | `#92836c` | Primary text / deep accents |
| `--clay` | `#a8957f` | Secondary surfaces |
| `--sand` | `#c8bba9` | Borders / muted |
| `--cream` | `#ecdcc4` | Cards / highlights |
| `--paper` | `#f7f1e9` | Background |

3D motifs are **topic-relevant** (paper sheets, document stacks, a knowledge
graph), built with React Three Fiber and animated with Framer Motion.

---

## Architecture

```
 Next.js (3D UI) ──HTTP/SSE──▶ FastAPI
                                  │
        ┌─────────────────────────┼──────────────────────────┐
   Ingestion                 Retrieval                   Generation
   load→split→embed→     FAISS + BM25 → RRF →         LCEL chain → LLM
   FAISS+BM25 index      rerank → CRAG/compress        (OpenAI/Claude/Ollama)
                                  │
                       Cache · RAGAS eval · Drift · Prometheus · Audit logs
```

See `PRD (1).md`, `TECHSTACK (1).md`, and `FEATURES_COMPLETE.md` for the full spec,
and [TODO.md](TODO.md) for the phased build plan and status.

---

## Make targets

```bash
make backend     # run FastAPI (reload)
make frontend    # run Next.js dev server
make test        # pytest + coverage
make ingest      # ingest data/uploads/
make eval        # run RAGAS evaluation
make lint        # ruff + mypy
make format      # black + isort
make docker-up   # docker compose up --build -d
```

## API surface

`/api/v1/documents` · `/api/v1/query` (+ `/stream`) · `/api/v1/conversations` ·
`/api/v1/evaluate` (+ `/drift`) · `/api/v1/settings` · `/api/v1/health` · `/metrics`.
Interactive docs at **`/docs`** (Swagger) and **`/redoc`**.
