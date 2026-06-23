# DocMind — Build TODO

> Atomic task list derived from `PRD (1).md`, `TECHSTACK (1).md`, and `FEATURES_COMPLETE.md`.
> **Rules:** Items *within a phase* have **no internal dependencies** — each can be built in isolation.
> Phases are ordered: a later phase may depend on an earlier one, never the reverse.
> After **every phase**, run the verification gate (syntax compile + ruff) before moving on.
>
> **Build decisions (locked):**
> - Frontend: **Next.js + React** (Three.js / R3F 3D, Framer/21st.dev components, palette, responsive) — replaces Streamlit.
> - Order: **Backend-first**, then animated frontend.
> - LLM/embeddings: **Local-first** (HuggingFace `all-MiniLM-L6-v2` + Ollama default, no API key required; OpenAI/Anthropic optional via `.env`).
>
> **Palette:** `#92836c` `#a8957f` `#c8bba9` `#ecdcc4` `#f7f1e9` (warm taupe → cream).

---

## PHASE 0 — Scaffolding & Config  *(no deps)*  ✅ DONE
- [x] P0.1 Create full directory tree (`app/`, `rag/`, `prompts/`, `streamlit_app/`→`frontend/`, `data/`, `evaluation/`, `logs/`, `tests/`)
- [x] P0.2 `requirements.txt` (pinned production deps)
- [x] P0.3 `requirements-dev.txt` (dev/test deps)
- [x] P0.4 `pyproject.toml` (black, isort, ruff, mypy config)
- [x] P0.5 `.env.example` (all variables, local-first safe defaults)
- [x] P0.6 `.gitignore`
- [x] P0.7 `.dockerignore`
- [x] P0.8 `Makefile` (dev, test, ingest, eval, lint, format, docker-up/down, clean)
- [x] P0.9 `app/config.py` — Pydantic `Settings` (local-first defaults)
- [x] **GATE 0:** ✅ config compiles + imports (local-first ollama/hf), ruff clean

## PHASE 1 — Pydantic Data Models  *(deps: P0)*  ✅ DONE
- [x] P1.1 `app/models/document.py` (DocumentRecord, DocumentStatus, ChunkPreview)
- [x] P1.2 `app/models/query.py` (QueryRequest, QueryResponse, SourceChunk, Feedback)
- [x] P1.3 `app/models/conversation.py` (ConversationSession, Message)
- [x] P1.4 `app/models/evaluation.py` (EvaluationRequest, EvaluationResult, DriftStatus)
- [x] P1.5 `app/models/settings.py` (SettingsResponse / patch model)
- [x] **GATE 1:** ✅ compile + import + ruff clean

## PHASE 2 — Prompts  *(no code deps)*  ✅ DONE
- [x] P2.1 `prompts/system_default.txt`
- [x] P2.2 `prompts/rag_qa.txt`
- [x] P2.3 `prompts/query_rewrite.txt`
- [x] P2.4 `prompts/hyde.txt`
- [x] P2.5 `prompts/compression.txt`
- [x] **GATE 2:** ✅ all 5 present & non-empty

## PHASE 3 — RAG Ingestion  *(deps: P0,P1)*  ✅ DONE
- [x] P3.1 `rag/ingestion/loader.py` (file-type router → LangChain loaders)
- [x] P3.2 `rag/ingestion/splitter.py` (recursive / semantic / parent strategies)
- [x] P3.3 `rag/ingestion/embedder.py` (OpenAI / HF factory, local-first auto-fallback)
- [x] P3.4 `rag/ingestion/ocr.py` (pytesseract fallback)
- [x] P3.5 `rag/ingestion/deduplicator.py` (sha256 hash dedup)
- [x] P3.6 `rag/ingestion/indexer.py` (FAISS + BM25 build/update/delete/persist)
- [x] P3.7 `rag/ingestion/pipeline.py` (orchestrate load→split→embed→index→save)
- [x] **GATE 3:** ✅ py_compile + ruff clean

## PHASE 4 — RAG Retrieval  *(deps: P0,P1,P3)*  ✅ DONE
- [x] P4.1 `rag/retrieval/faiss_retriever.py`
- [x] P4.2 `rag/retrieval/bm25_retriever.py`
- [x] P4.3 `rag/retrieval/hybrid.py` (Ensemble + RRF)
- [x] P4.4 `rag/retrieval/reranker.py` (cross-encoder)
- [x] P4.5 `rag/retrieval/query_expansion.py` (MultiQuery + HyDE)
- [x] P4.6 `rag/retrieval/compressor.py` (LLMChainExtractor)
- [x] P4.7 `rag/retrieval/crag.py` (corrective RAG relevance eval)
- [x] P4.8 `rag/retrieval/self_rag.py` (reflection loop)
- [x] P4.9 `rag/retrieval/factory.py` (build retriever from config)
- [x] **GATE 4:** ✅ py_compile + ruff clean

## PHASE 5 — RAG Generation  *(deps: P0,P1,P2)*  ✅ DONE
- [x] P5.1 `rag/generation/llm_factory.py` (OpenAI/Anthropic/Ollama + fallback chain)
- [x] P5.2 `rag/generation/memory.py` (buffer-window + summary)
- [x] P5.3 `rag/generation/cost_tracker.py` (token→cost per provider)
- [x] P5.4 `rag/generation/streaming.py` (SSE token helpers)
- [x] P5.5 `rag/generation/chain.py` (LCEL RAG chain)
- [x] **GATE 5:** ✅ py_compile + ruff clean

## PHASE 6 — Cache, Evaluation, Security  *(deps: P0,P1)*  ✅ DONE
- [x] P6.1 `rag/cache/semantic_cache.py` (InMemory / Redis)
- [x] P6.2 `rag/evaluation/ragas_eval.py`
- [x] P6.3 `rag/evaluation/testset_gen.py`
- [x] P6.4 `rag/evaluation/drift.py` (Evidently)
- [x] P6.5 `rag/evaluation/metrics.py` (Recall@k, NDCG, MRR)
- [x] P6.6 `rag/security/pii_redactor.py`
- [x] P6.7 `rag/security/access_control.py`
- [x] **GATE 6:** ✅ py_compile + ruff clean

## PHASE 7 — FastAPI Backend  *(deps: all rag + models)*  ✅ DONE
- [x] P7.1 `app/middleware/logging.py`
- [x] P7.2 `app/middleware/timing.py`
- [x] P7.3 `app/middleware/api_key.py`
- [x] P7.4 `app/middleware/rate_limiter.py`
- [x] P7.5 `app/dependencies.py` (Depends providers: store, llm, embeddings)
- [x] P7.6 `app/api/health.py` (health + Prometheus /metrics)
- [x] P7.7 `app/api/documents.py`
- [x] P7.8 `app/api/query.py` (+ SSE stream)
- [x] P7.9 `app/api/conversations.py`
- [x] P7.10 `app/api/evaluate.py`
- [x] P7.11 `app/api/settings.py`
- [x] P7.12 `app/main.py` (app, lifespan, middleware, router registration)
- [x] **GATE 7:** ✅ real import test — app builds, **24 endpoints** registered, ruff clean
      *(added langchain 0.2/1.x import shims for cross-version resilience)*

## PHASE 8 — Tests  *(deps: P7)*  ✅ DONE
- [x] P8.1 `tests/conftest.py` (fixtures: test docs, fake LLM, temp store)
- [x] P8.2 `tests/test_ingestion.py`
- [x] P8.3 `tests/test_retrieval.py`
- [x] P8.4 `tests/test_chain.py`
- [x] P8.5 `tests/test_cache.py`
- [x] P8.6 `tests/test_evaluation.py`
- [x] P8.7 `tests/test_api/test_documents_api.py`
- [x] P8.8 `tests/test_api/test_query_api.py`
- [x] P8.9 `tests/test_api/test_evaluate_api.py`
- [x] **GATE 8:** ✅ **37 tests pass** (live FastAPI TestClient + unit), ruff clean

## PHASE 9 — Docker & Deploy  *(deps: P7)*  ✅ DONE
- [x] P9.1 `Dockerfile` (python:3.11-slim + tesseract/poppler) + `frontend.Dockerfile`
- [x] P9.2 `docker-compose.yml` (backend, frontend, redis, ollama)
- [x] P9.3 `README.md` (setup, run, palette, architecture)
- [x] **GATE 9:** ✅ `docker compose config` validates

## PHASE 10 — Frontend Scaffold (Next.js)  *(deps: P7 API contract)*  ✅ DONE
- [x] P10.1 `create-next-app` (TS, App Router, Tailwind, src dir) in `frontend/`
- [x] P10.2 Tailwind theme = palette tokens; editorial type; flat ink-on-paper (no glass/gradient)
- [x] P10.3 API client (`lib/api.ts`) typed against FastAPI endpoints (+ SSE stream parser)
- [x] P10.4 three / R3F / drei / framer-motion installed (React-18-compatible pins)
- [x] **GATE 10:** ✅ `npm run build` compiles

## PHASE 11 — Frontend 3D Components & Pages  *(deps: P10)*  ✅ DONE
- [x] P11.1 3D scenes: `HeroScene` (floating document pages), `KnowledgeGraphScene` (embedding graph)
- [x] P11.2 Framer components: ImageScroller, ProjectHoverCard, TextArrowCTA, CircleExpandButton, RunaroundText, Reveal
- [x] P11.3 Landing/Hero page (3D document stack, scroller, run-around, CTA)
- [x] P11.4 Chat page (SSE streaming, citations, confidence badge, retrieval controls)
- [x] P11.5 Documents page (drag-drop upload, table, status)
- [x] P11.6 Evaluation page (RAGAS metric gauges, drift card)
- [x] P11.7 Settings page (live PATCH to backend)
- [x] P11.8 About page (pipeline walkthrough, 3D, tech stack)
- [x] P11.9 Responsive pass (mobile nav, md/lg grids throughout)
- [x] **GATE 11:** ✅ `npm run build` + type-check + lint clean (6 pages)

## PHASE 12 — Integration & Polish  *(deps: P7,P11)*  ✅ DONE
- [x] P12.1 Frontend wired to live backend; SSE stream parser in `lib/api.ts`
- [x] P12.2 Error / empty / loading states on every page (unreachable-backend, 409, no-docs)
- [x] P12.3 Responsive + reduced-motion + a11y (prefers-reduced-motion, focus styles, mobile nav)
- [x] P12.4 End-to-end smoke: backend boots, all frontend-facing endpoints 200/409, all 6 pages serve 200
- [x] **GATE 12:** ✅ full stack boots; backend ↔ frontend contract verified live

---

## Verification summary
- **Backend:** app imports, **24 endpoints** registered, **37 tests pass**, ruff clean, `docker compose config` valid, boots under uvicorn and answers health/settings/documents/stats/query(409)/docs.
- **Frontend:** `npm run build` clean (type-check + lint), 6 pages, all serve 200 under `next start`, wired to the API.
- **Not run locally** (documented): real LLM/embedding inference needs the multi-GB torch/sentence-transformers stack or Ollama — runs in Docker (`python:3.11-slim` + pinned deps) or with `ollama pull llama3`. The pinned `requirements.txt` targets Python 3.11; local verification used Python 3.13 with API-compatible shims for langchain 1.x.
</content>
</invoke>
