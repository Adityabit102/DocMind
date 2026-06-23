# DocMind — Complete & Exhaustive Feature List
## RAG-Powered Legal / Medical Document Q&A System

> **Purpose:** This document catalogues EVERY feature — big, small, significant, insignificant — that must be implemented in the DocMind project. Built from the PRD, TECHSTACK, and comprehensive 2025–2026 web research on production RAG systems. If it is not in this list, it does not need to be added.

---

## SECTION 1 · DOCUMENT INGESTION PIPELINE

### 1.1 Supported File Formats
- [ ] PDF with text layer (via `PyPDFLoader`)
- [ ] Scanned PDF with no text layer → auto-detect and OCR fallback (via `pytesseract` + `pdf2image`)
- [ ] DOCX (via `Docx2txtLoader` + `python-docx`)
- [ ] TXT files
- [ ] Markdown `.md` files
- [ ] CSV (table-aware: each row = one chunk)
- [ ] HTML (tag-stripped via `UnstructuredHTMLLoader`)
- [ ] Images embedded inside PDFs → LLM vision captioning (toggle in settings)

### 1.2 Pre-Processing & Cleaning
- [ ] Encoding detection (`chardet`) and normalisation to UTF-8
- [ ] Whitespace collapse and ligature normalisation
- [ ] Header and footer stripping
- [ ] Table detection and preservation as structured text
- [ ] Multi-column PDF layout handling (`UnstructuredPDFLoader`)
- [ ] Metadata extraction: title, author, page count, word count, creation date
- [ ] Duplicate file detection (hash-based) on upload
- [ ] Provenance tracking per chunk: source file, page number, chunk index, character offset, version

### 1.3 OCR Subsystem
- [ ] Auto-trigger OCR when PyPDF extracts < 50 chars/page
- [ ] Page-level OCR via `pytesseract` (Tesseract engine)
- [ ] PDF → image conversion via `pdf2image` before OCR
- [ ] Image preprocessing via `Pillow` (deskew, denoise)
- [ ] User-facing notification when OCR is triggered

### 1.4 Chunking Strategies (all selectable from UI)
- [ ] **RecursiveCharacterTextSplitter** (default): chunk_size=1000, overlap=200, separators: `\n\n → \n → ". " → " " → ""`
- [ ] **SemanticChunker**: splits at embedding distance breakpoints between sentences
- [ ] **ParentDocumentRetriever** pattern: small child chunks for retrieval, large parent chunks for context
- [ ] **Table-aware chunking**: each table row or cell as independent chunk
- [ ] **Hierarchical chunking**: summary-level chunks (H1/H2) + detail-level chunks (paragraph)
- [ ] `add_start_index=True` on all splitters: preserves character offset for citation
- [ ] Chunk metadata stamps: source file, page number, chunk index, char offset, ingestion timestamp
- [ ] Configurable chunk_size and chunk_overlap from settings panel (200–4000)

### 1.5 Embedding Models (all switchable from UI)
- [ ] **OpenAI `text-embedding-3-small`** (primary): 1536-dim, best cost/quality ratio
- [ ] **`all-MiniLM-L6-v2`** (local fallback): 384-dim, no API key required, CPU-only
- [ ] **`BAAI/bge-large-en-v1.5`** (high-accuracy): 1024-dim, best open-source English embeddings
- [ ] Auto-fallback: if `OPENAI_API_KEY` not set → switch to `all-MiniLM-L6-v2`
- [ ] Batch embedding with exponential backoff retry on rate-limit errors
- [ ] Embedding model recorded in document metadata for index consistency checks

### 1.6 Vector Indexing
- [ ] **FAISS `IndexFlatL2`** for ≤ 100k chunks (exact, no config needed)
- [ ] **FAISS `IndexIVFFlat`** for > 100k chunks (approximate, faster at scale)
- [ ] FAISS index persisted to disk at `data/faiss_index/` (`index.faiss` + `index.pkl`)
- [ ] **BM25 sparse index** maintained in parallel (`rank_bm25`), persisted at `data/faiss_index/bm25.pkl`
- [ ] Automatic index rebuild on document add or delete
- [ ] FAISS auto-rebuild from stored chunk metadata if index file is missing or corrupt
- [ ] Async ingestion pipeline: UI shows per-file progress bar, API returns job ID for polling

---

## SECTION 2 · RETRIEVAL PIPELINE

### 2.1 Pre-Retrieval: Query Processing
- [ ] **Query rewriting**: LLM rewrites vague/short queries into explicit search-friendly form
- [ ] **Multi-query expansion** (`MultiQueryRetriever`): LLM generates 3 semantically diverse versions; results union-merged with deduplication
- [ ] **HyDE (Hypothetical Document Embeddings)**: LLM generates a hypothetical answer → embedded as query vector → improves recall on abstract questions (toggle)
- [ ] **Conversation history fusion**: last N turns summarised and injected into query context
- [ ] Query length check: warn user if query is < 3 words

### 2.2 Retrieval Modes (selectable per query from sidebar)
- [ ] **Similarity mode**: FAISS cosine similarity, top-k candidates
- [ ] **MMR mode** (Maximal Marginal Relevance): balances relevance and diversity (`lambda_mult` configurable)
- [ ] **Hybrid mode** (default): dense FAISS + sparse BM25 fused via RRF

### 2.3 Hybrid Retrieval (Ensemble)
- [ ] **Dense retrieval**: FAISS cosine similarity, top-20 candidates
- [ ] **Sparse retrieval**: BM25 on raw text, top-20 candidates
- [ ] **RRF Fusion** (Reciprocal Rank Fusion): merge and re-score both lists into unified top-30
- [ ] Ensemble weights configurable: default [0.6 dense, 0.4 BM25], tunable from settings

### 2.4 Metadata Filtering (pre-retrieval)
- [ ] Filter by document name / filename
- [ ] Filter by user-assigned tag (e.g., "legal", "research")
- [ ] Filter by page range (start page – end page)
- [ ] Scope selector in sidebar: "All documents" or named subset

### 2.5 Post-Retrieval Processing
- [ ] **Cross-encoder reranking** (`cross-encoder/ms-marco-MiniLM-L-6-v2`): reranks top-30 → top-5
- [ ] **Context compression** (`LLMChainExtractor`): removes irrelevant sentences from each chunk → reduces token count, improves signal (toggle)
- [ ] **Parent document expansion**: reranked child chunks retrieve their parent chunk for richer context
- [ ] **Diversity deduplication**: deduplicate nearly identical chunks (> 0.95 cosine similarity between retrieved chunks)
- [ ] **Corrective RAG (CRAG)** relevance evaluator: lightweight classifier checks retrieved docs; if relevance score < threshold, triggers re-query with refined terms
- [ ] **Self-RAG reflection**: after initial answer generation, model self-critiques whether answer is grounded; re-retrieves if not
- [ ] **Zero-chunk guardrail**: if retrieval returns 0 relevant chunks → return "No relevant content found" without making LLM call

---

## SECTION 3 · GENERATION PIPELINE

### 3.1 Prompt Engineering
- [ ] System prompt: persona, grounding instruction ("answer only from the provided context"), citation format, fallback phrase ("I don't have enough information in the provided documents")
- [ ] Context formatted as: `[Source: filename, Page: N] chunk text`
- [ ] Chat history injected as alternating user/assistant turns via `MessagesPlaceholder`
- [ ] Prompt templates stored as versioned `.txt` files in `prompts/` directory:
  - `prompts/system_default.txt`
  - `prompts/rag_qa.txt` (main RAG Q&A prompt)
  - `prompts/query_rewrite.txt`
  - `prompts/hyde.txt`
  - `prompts/compression.txt`
- [ ] Custom system prompt editable by user from Settings page
- [ ] Context window size awareness: truncate chunks from tail if context exceeds model limit; show warning banner
- [ ] Grounding instruction: model explicitly told to say "I don't know" if context is insufficient

### 3.2 LLM Providers (all selectable from UI)
- [ ] **OpenAI GPT-4o-mini** (default, cheapest with strong accuracy)
- [ ] **OpenAI GPT-4o** (higher accuracy, higher cost)
- [ ] **Anthropic Claude** (via `langchain-anthropic`): `claude-3-5-sonnet-20241022`, `claude-3-haiku-20240307`
- [ ] **Ollama local** (zero API cost): `llama3`, `mistral`, `phi3`, `gemma2` — pull command: `ollama pull llama3`
- [ ] Provider fallback chain: OpenAI → Anthropic → Ollama
- [ ] Temperature configurable (0.0–1.0, default 0.0 for factual grounding)
- [ ] Max tokens configurable (128–8192, default 1024)

### 3.3 LCEL Chain Composition
- [ ] LCEL declarative RAG chain: `retriever | format_docs_with_citations | prompt_template | llm | StrOutputParser()`
- [ ] Parallel context + question passthrough via `RunnablePassthrough`
- [ ] `rag_chain.stream()` for real-time token streaming
- [ ] `await rag_chain.ainvoke()` for async non-streaming queries
- [ ] Exponential backoff retry wrapper on all LLM calls (`tenacity`, max 3 retries)

### 3.4 Conversation Memory
- [ ] `ConversationBufferWindowMemory(k=5)`: last 5 turns held in window
- [ ] `ConversationSummaryMemory(llm=llm)`: older turns summarised to save context
- [ ] Memory cleared on "New Chat" button click
- [ ] Session state persisted across Streamlit reruns via `st.session_state`
- [ ] Multi-turn follow-up support ("What did you mean by X?", "Expand on point 2")

### 3.5 Output Generation
- [ ] Streaming response via SSE (Server-Sent Events) — token-by-token rendering
- [ ] Final answer + ordered list of source chunks (file, page, chunk text excerpt, relevance score)
- [ ] Confidence estimation: average reranker score of top-k chunks → `High ●` / `Medium ●` / `Low ●` badge
- [ ] Token usage logged per query (prompt tokens, completion tokens)
- [ ] Cost per query calculated and displayed (token count × model price per token)

---

## SECTION 4 · SEMANTIC CACHING

- [ ] **Development cache**: `InMemorySemanticCache` (LangChain built-in), cosine threshold 0.95, resets on restart
- [ ] **Production cache**: `RedisSemanticCache` (redis-py + langchain-redis), persists across restarts
- [ ] Cache TTL configurable (default 86400s = 24h)
- [ ] Cache hit → response returned in < 200ms without LLM call
- [ ] Cache hit / miss logged for observability
- [ ] Cache stats (hit rate, size, memory used) visible in admin panel
- [ ] Cache can be manually cleared from admin panel / settings page
- [ ] Cache enable/disable toggle in sidebar

---

## SECTION 5 · EVALUATION & QUALITY MEASUREMENT

### 5.1 RAGAS Metrics Dashboard
- [ ] **Faithfulness**: is every claim in the answer supported by the retrieved context? (0–1)
- [ ] **Answer Relevancy**: does the answer actually address the question? (0–1)
- [ ] **Context Precision**: are retrieved chunks genuinely relevant? (0–1)
- [ ] **Context Recall**: does the context cover the ground-truth answer? (0–1)
- [ ] Metric cards displayed with `st.metric` (current score + delta vs last run)
- [ ] Line chart: metric scores over evaluation runs (trend over time)
- [ ] Per-query breakdown table (downloadable as CSV)
- [ ] Run evaluation on-demand with a "Run Evaluation" button
- [ ] Scheduled/auto evaluation configurable (e.g., weekly)

### 5.2 Synthetic Test Set Generation
- [ ] `ragas.testset.generator.TestsetGenerator.with_openai()` generates synthetic Q&A pairs
- [ ] Test set size configurable (default 50 questions)
- [ ] Test sets stored in `evaluation/testsets/testset_<date>.csv`
- [ ] Evaluation results stored in `evaluation/results/eval_<date>.json` + `.csv`
- [ ] Export all evaluation metrics to CSV (button on Evaluation page)

### 5.3 LLM-as-Judge / Grounding Score
- [ ] Grounding score: measures how much the answer derives from retrieved docs vs model memory
- [ ] Per-response faithfulness check: highlight any sentence in the answer not traceable to a source chunk
- [ ] Unsupported Sentence Ratio (USR) tracked per evaluation run

### 5.4 Retrieval Quality Metrics
- [ ] Recall@k: what fraction of relevant chunks are retrieved in top-k
- [ ] NDCG (Normalized Discounted Cumulative Gain) on retrieval ranking
- [ ] Mean Reciprocal Rank (MRR) of first relevant chunk
- [ ] Average chunk relevance score distribution (histogram)

### 5.5 Drift Detection
- [ ] Evidently AI embedding distribution drift report: compare last 7 days vs baseline
- [ ] Alert triggered when drift score exceeds configurable threshold
- [ ] Drift report saved and viewable in Evaluation page
- [ ] `GET /api/v1/evaluate/drift` endpoint returns current drift status

---

## SECTION 6 · OBSERVABILITY & LOGGING

### 6.1 Structured Logging
- [ ] JSON-structured logs via `python-json-logger`
- [ ] Every query logs: query text, rewritten query, retrieval mode, chunks retrieved, chunks after rerank, top chunk score, LLM model, prompt tokens, completion tokens, cost USD, latency ms, cache hit, timestamp, session ID
- [ ] PII scrubbing option: query text logging can be disabled via `ENABLE_QUERY_LOGGING=false`
- [ ] Log level configurable (`LOG_LEVEL` env var): DEBUG / INFO / WARNING / ERROR
- [ ] Logs written to `logs/app.log` + stdout (for Docker log aggregation)

### 6.2 Prometheus Metrics
- [ ] `rag_queries_total` Counter (labels: model, cache_hit, retrieval_mode)
- [ ] `rag_query_latency_seconds` Histogram (p50, p95 percentiles)
- [ ] `rag_documents_total` Gauge (indexed document count)
- [ ] `rag_chunks_total` Gauge (total indexed chunk count)
- [ ] `rag_ingestion_duration_seconds` Histogram
- [ ] `rag_cache_hit_ratio` Gauge
- [ ] `rag_errors_total` Counter (labels: error_type)
- [ ] Exposed at `GET /metrics` (scrape-able by Prometheus / Grafana Cloud)

### 6.3 LangSmith Tracing (Optional)
- [ ] Full LCEL chain traces via `LANGSMITH_API_KEY` + `LANGCHAIN_TRACING_V2=true`
- [ ] Per-step latency breakdown: retrieval, reranking, compression, generation
- [ ] Token usage and cost per step visible in LangSmith dashboard
- [ ] Enabled/disabled via env vars — zero code change required

### 6.4 Admin Query Log Viewer
- [ ] Query log table in admin panel: query, latency, tokens, cost, cache hit, timestamp
- [ ] Latency histogram in UI (chart of p50/p95 over time)
- [ ] Token usage over time chart
- [ ] Cost over time chart (cumulative and per-session)
- [ ] Export query logs as CSV

### 6.5 Audit Trail (Security)
- [ ] Tamper-evident append-only audit logs: authenticated user identity, timestamp, request ID, sanitised query, retrieval filters, document IDs retrieved, model version, generated response
- [ ] Audit log retention policy configurable
- [ ] Each answer traceable back to specific source documents (required for legal/medical compliance)

---

## SECTION 7 · SECURITY

### 7.1 Input Security
- [ ] Input sanitisation on all user text fields (prevent prompt injection)
- [ ] Query length hard cap (configurable, default 2000 chars)
- [ ] File type validation on upload (reject unsupported formats with clear error)
- [ ] File size limit on upload (configurable, default 50MB)
- [ ] Malicious file detection: reject files with suspicious content patterns

### 7.2 API Security
- [ ] Optional API key authentication via `X-API-Key` header (`API_KEY_AUTH_ENABLED` toggle)
- [ ] API keys stored in `.env` only — never in code, logs, or responses
- [ ] CORS restricted to Streamlit frontend origin (`CORS_ORIGINS`)
- [ ] Rate limiting per session/IP: configurable requests per minute (`API_RATE_LIMIT`)
- [ ] Automatic temporary ban after exceeding rate limit threshold
- [ ] HTTPS-ready (TLS termination via reverse proxy in production)

### 7.3 Data Security
- [ ] Uploaded documents stored in `data/uploads/`, NOT served as static files
- [ ] Document-level access control: document classification levels (Public / Internal / Confidential)
- [ ] Role-based document filtering: admin sees all, user sees only permitted docs
- [ ] Chunk-level security validation: poisoned/injected content detection
- [ ] PII pattern detection and redaction in responses (toggle)
- [ ] Sensitive information filtering on output (configurable blocklist)
- [ ] No API keys or secrets ever logged (secret scrubbing in log middleware)

### 7.4 Retrieval Security
- [ ] Security trimming at retrieval time: only surface chunks the requesting user is authorised to see
- [ ] Provenance-per-chunk: source system, document ID, owner, classification, ingestion date — all stored for auditability

---

## SECTION 8 · REST API (FastAPI)

### 8.1 Document Endpoints
- [ ] `POST /api/v1/documents/upload` — upload and ingest one or multiple files
- [ ] `GET /api/v1/documents` — list all indexed documents with metadata
- [ ] `GET /api/v1/documents/{id}` — get single document record
- [ ] `DELETE /api/v1/documents/{id}` — delete document and trigger re-index
- [ ] `GET /api/v1/documents/{id}/chunks` — list all chunks for a document (chunk preview)
- [ ] `POST /api/v1/documents/{id}/reindex` — re-index a single document
- [ ] `GET /api/v1/documents/{id}/download` — download original uploaded file
- [ ] `PATCH /api/v1/documents/{id}/tags` — update document tags

### 8.2 Query Endpoints
- [ ] `POST /api/v1/query` — single Q&A query (JSON response with answer + sources)
- [ ] `POST /api/v1/query/stream` — streaming Q&A via SSE (token-by-token)
- [ ] `POST /api/v1/query/regenerate` — regenerate answer with different retrieval strategy
- [ ] `POST /api/v1/feedback` — submit thumbs up/down + comment for an answer

### 8.3 Conversation Endpoints
- [ ] `GET /api/v1/conversations` — list all conversation sessions
- [ ] `POST /api/v1/conversations` — create new conversation session
- [ ] `GET /api/v1/conversations/{id}` — get full conversation history
- [ ] `DELETE /api/v1/conversations/{id}` — delete conversation
- [ ] `GET /api/v1/conversations/{id}/export` — export as PDF or Markdown

### 8.4 Evaluation Endpoints
- [ ] `POST /api/v1/evaluate` — run RAGAS evaluation on current index
- [ ] `GET /api/v1/evaluate/results` — get evaluation scores
- [ ] `GET /api/v1/evaluate/export` — download scores as CSV
- [ ] `GET /api/v1/evaluate/drift` — current drift detection status

### 8.5 System Endpoints
- [ ] `GET /api/v1/health` — liveness probe (Docker health check)
- [ ] `GET /metrics` — Prometheus metrics scrape endpoint
- [ ] `GET /api/v1/settings` — get current config
- [ ] `PATCH /api/v1/settings` — update config (LLM, retrieval mode, etc.)
- [ ] `DELETE /api/v1/cache` — clear semantic cache
- [ ] `GET /api/v1/admin/stats` — storage usage, doc count, chunk count, query count

### 8.6 API Metadata
- [ ] Swagger UI auto-generated at `/docs`
- [ ] ReDoc alternative at `/redoc`
- [ ] All request/response models validated with Pydantic v2
- [ ] `BackgroundTasks` for async document ingestion (non-blocking upload)
- [ ] `StreamingResponse` with `media_type="text/event-stream"` for SSE
- [ ] Request logging middleware (logs method, path, status, latency)
- [ ] Response timing middleware (injects `X-Process-Time` header)
- [ ] Lifespan events: load FAISS index on startup, flush cache on shutdown

---

## SECTION 9 · FRONTEND (STREAMLIT)

### 9.1 Multi-Page App Structure
- [ ] `app.py` — Page 0: Main Chat Interface (default landing page)
- [ ] `pages/1_Documents.py` — Document Upload & Management
- [ ] `pages/2_Evaluation.py` — RAGAS Evaluation Dashboard
- [ ] `pages/3_Settings.py` — Configuration & Model Selector
- [ ] `pages/4_About.py` — Tech Stack, README, OpenAPI link

### 9.2 Chat Page — Main Interface
**Sidebar (left panel):**
- [ ] Document scope filter (multi-select: all docs or subset)
- [ ] Tag filter (filter Q&A to documents with specific tags)
- [ ] Retrieval mode selector (Similarity / MMR / Hybrid)
- [ ] Top-k slider (1–50, default 20)
- [ ] Temperature slider (0.0–1.0)
- [ ] Toggle: Query Expansion (on/off)
- [ ] Toggle: HyDE (on/off)
- [ ] Toggle: Reranking (on/off)
- [ ] Toggle: Context Compression (on/off)
- [ ] Toggle: Semantic Cache (on/off)
- [ ] Conversation history list (clickable sessions)
- [ ] "New Chat" button (clears memory)
- [ ] "Export Chat" button (PDF or Markdown)

**Main area (right panel):**
- [ ] `st.chat_message` bubbles for user and assistant turns
- [ ] `st.write_stream` for real-time token streaming
- [ ] Collapsible source expanders per citation: `[Page N · filename]` → shows chunk text (highlighted) + relevance score badge
- [ ] Confidence badge on each answer: `High ●`, `Medium ●`, `Low ●`
- [ ] Thumbs up / thumbs down buttons per answer (with optional comment box)
- [ ] Copy-to-clipboard button per answer (one click)
- [ ] Regenerate button per answer (re-run with different retrieval strategy)
- [ ] Session cost display: "Cost this session: $0.003"
- [ ] Session token counter: "Tokens used: 1,204"
- [ ] "Clear conversation" button

### 9.3 Documents Page
- [ ] Drag-and-drop multi-file uploader (`st.file_uploader(accept_multiple_files=True)`)
- [ ] Per-file upload progress bar (`st.progress` + `st.status`)
- [ ] Indexed documents table (`st.data_editor`): name, type, pages, chunks, size, indexed date, tags, status (queued / processing / indexed / failed), actions
- [ ] Tag input field per document (comma-separated)
- [ ] Delete button per document (with confirmation dialog)
- [ ] Re-index button (per document or "Re-index All")
- [ ] Download original file button per document
- [ ] Chunk preview modal: all chunks for selected doc with metadata (page, offset, char count)
- [ ] Storage usage display (total uploaded files size, total indexed chunks)
- [ ] Filter/search bar above document table

### 9.4 Evaluation Page
- [ ] "Run Evaluation" button → triggers RAGAS on current index
- [ ] Loading spinner during evaluation run
- [ ] Metric cards: Faithfulness, Answer Relevancy, Context Precision, Context Recall (`st.metric`)
- [ ] Delta display vs previous run on each metric card
- [ ] Line chart: metric trends over evaluation runs (`st.line_chart`)
- [ ] Per-query breakdown table (question, answer, faithfulness, relevancy — sortable)
- [ ] "Export to CSV" button (downloads full evaluation results)
- [ ] Drift detection status card (OK / Drift Detected)
- [ ] "Run Drift Check" button

### 9.5 Settings Page
- [ ] LLM provider selector (OpenAI / Anthropic / Ollama)
- [ ] LLM model selector (model names per provider)
- [ ] Embedding model selector (OpenAI / HuggingFace options)
- [ ] Chunk size input
- [ ] Chunk overlap input
- [ ] Chunking strategy selector (Recursive / Semantic / Parent-Document)
- [ ] Context window size selector
- [ ] Max tokens slider
- [ ] System prompt text area (editable, with reset-to-default button)
- [ ] API key auth toggle
- [ ] Cache backend selector (Memory / Redis)
- [ ] Cache TTL input
- [ ] Log level selector
- [ ] "Save Settings" button → persists to `.env` + `session_state`
- [ ] "Reset to Defaults" button

### 9.6 About Page
- [ ] README display with project description
- [ ] Tech stack table (all libraries + versions)
- [ ] Links: GitHub repo, OpenAPI docs at `/docs`, LangSmith project
- [ ] RAGAS scores table (last eval scores prominently shown)
- [ ] Architecture diagram or flowchart

### 9.7 UI/UX Details
- [ ] `st.toast` for non-blocking success/error notifications
- [ ] `st.expander` for collapsible source citation blocks
- [ ] `st.tabs` to organise chat + sources side by side (optional view)
- [ ] `st.cache_resource` to cache FAISS index and LLM client across Streamlit sessions
- [ ] `streamlit-extras` for: copy-to-clipboard, metric card enhancements
- [ ] Dark/light theme support (Streamlit default theme)
- [ ] Responsive layout (usable on tablet/mobile via Streamlit's responsive CSS)

---

## SECTION 10 · DATA MODELS

### 10.1 Document Record
```json
{
  "id": "uuid",
  "filename": "report.pdf",
  "file_type": "pdf",
  "file_size_bytes": 204800,
  "page_count": 42,
  "word_count": 18500,
  "chunk_count": 87,
  "tags": ["legal", "2024"],
  "indexed_at": "2025-06-20T10:30:00Z",
  "embedding_model": "text-embedding-3-small",
  "status": "indexed",
  "classification": "internal",
  "hash": "sha256:abc123..."
}
```

### 10.2 Query Log Record
```json
{
  "query_id": "uuid",
  "session_id": "uuid",
  "original_query": "What is the revenue?",
  "rewritten_query": "What was the total revenue figure reported?",
  "hyde_query": "The total revenue reported was...",
  "retrieval_mode": "hybrid",
  "chunks_retrieved": 20,
  "chunks_after_rerank": 5,
  "top_chunk_score": 0.91,
  "confidence_level": "high",
  "llm_model": "gpt-4o-mini",
  "prompt_tokens": 1450,
  "completion_tokens": 280,
  "cost_usd": 0.00028,
  "latency_ms": 2340,
  "cache_hit": false,
  "faithfulness_score": 0.93,
  "timestamp": "2025-06-20T10:35:00Z"
}
```

### 10.3 Evaluation Result
```json
{
  "run_id": "uuid",
  "timestamp": "2025-06-20T11:00:00Z",
  "question_count": 50,
  "faithfulness": 0.87,
  "answer_relevancy": 0.91,
  "context_precision": 0.84,
  "context_recall": 0.79,
  "unsupported_sentence_ratio": 0.07,
  "per_question": [...]
}
```

### 10.4 Source Chunk
```json
{
  "chunk_id": "uuid",
  "document_id": "uuid",
  "filename": "report.pdf",
  "page_number": 3,
  "chunk_index": 12,
  "char_offset": 4820,
  "text": "...",
  "relevance_score": 0.91,
  "reranker_score": 0.87,
  "embedding_model": "text-embedding-3-small"
}
```

---

## SECTION 11 · ERROR HANDLING & EDGE CASES

| Scenario | Behaviour |
|----------|-----------|
| Document has no extractable text (scanned) | OCR fallback triggered; user notified via toast |
| LLM API rate limit hit | Exponential backoff, max 3 retries; error shown if all fail |
| Query retrieves 0 chunks | "No relevant content found" — no LLM call made |
| LLM returns answer not grounded in context | Low confidence badge; `Self-RAG` re-retrieval attempted |
| FAISS index file missing / corrupt | Auto-rebuild from stored chunk metadata |
| Upload of unsupported file type | Clear error message listing supported types |
| Context window exceeded | Tail chunks truncated; user notified via warning banner |
| Embedding API unavailable | Auto-switch to local `all-MiniLM-L6-v2` fallback |
| Redis cache unavailable | Auto-fallback to in-memory cache; warning logged |
| LLM provider unavailable | Provider fallback chain: OpenAI → Anthropic → Ollama |
| Duplicate file uploaded | Hash-based detection; show warning, skip re-index |
| Oversized file upload | Reject with size limit message before ingestion starts |

---

## SECTION 12 · CONFIGURATION & ENVIRONMENT

### 12.1 `.env` Variables (Complete List)
```dotenv
# LLM
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OLLAMA_BASE_URL=http://ollama:11434
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=1024

# Embeddings
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small

# Retrieval
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
CHUNK_STRATEGY=recursive
RETRIEVAL_K=20
RERANKER_TOP_K=5
RETRIEVAL_MODE=hybrid
ENSEMBLE_DENSE_WEIGHT=0.6
ENSEMBLE_BM25_WEIGHT=0.4
ENABLE_QUERY_EXPANSION=true
ENABLE_HYDE=false
ENABLE_RERANKING=true
ENABLE_CONTEXT_COMPRESSION=false
ENABLE_CRAG=true
ENABLE_SELF_RAG=false

# Cache
CACHE_BACKEND=memory
REDIS_URL=redis://redis:6379
CACHE_TTL_SECONDS=86400
CACHE_SCORE_THRESHOLD=0.95

# Observability
LANGSMITH_API_KEY=
LANGCHAIN_TRACING_V2=false
LANGCHAIN_PROJECT=docmind
LOG_LEVEL=INFO
ENABLE_QUERY_LOGGING=true

# Security
API_KEY_AUTH_ENABLED=false
API_KEY=your-secret-key
CORS_ORIGINS=http://localhost:8501
API_RATE_LIMIT=60
MAX_UPLOAD_SIZE_MB=50
ENABLE_PII_REDACTION=false

# Storage
UPLOAD_DIR=data/uploads
FAISS_INDEX_DIR=data/faiss_index
EVAL_RESULTS_DIR=evaluation/results
METADATA_FILE=data/metadata.json

# Evaluation
RAGAS_TEST_SIZE=50
DRIFT_ALERT_THRESHOLD=0.15
```

### 12.2 Pydantic Settings Model
- [ ] All env vars parsed through `pydantic-settings` `BaseSettings` class in `app/config.py`
- [ ] Type validation on all settings at startup
- [ ] No hard-coded paths — everything configurable via environment

---

## SECTION 13 · DOCKER & DEPLOYMENT

### 13.1 Docker Compose Services
- [ ] `backend` service: FastAPI + Uvicorn (`port 8000`)
- [ ] `frontend` service: Streamlit (`port 8501`)
- [ ] `redis` service: Redis 7.2-alpine with AOF persistence (`port 6379`)
- [ ] `ollama` service: local LLM inference (`port 11434`) — optional profile: `docker compose --profile local-llm up`

### 13.2 Docker Features
- [ ] `Dockerfile`: `python:3.11-slim`, system deps: `tesseract-ocr`, `poppler-utils`, `libgl1`
- [ ] Health check on backend: `curl /api/v1/health` every 30s
- [ ] Volume mounts: `./data:/app/data` (persists FAISS index + uploads across container restarts)
- [ ] Named volumes: `redis_data`, `ollama_data`
- [ ] `.dockerignore`: excludes `.env`, `data/`, `logs/`, `__pycache__`, `.git`

### 13.3 One-Command Deploy
- [ ] `docker compose up --build -d` — full stack running in one command
- [ ] `docker compose down` — tear down
- [ ] `.env.example` with all variables documented and safe defaults

---

## SECTION 14 · CODE QUALITY & TOOLING

### 14.1 Linting & Formatting
- [ ] `ruff` (fast linter, replaces flake8 + pylint)
- [ ] `black` (opinionated code formatter)
- [ ] `isort` (import sorting)
- [ ] `mypy` (static type checking, mypy-clean)
- [ ] 100% type-annotated Python throughout
- [ ] Docstrings on all public functions

### 14.2 Pre-commit Hooks
- [ ] `.pre-commit-config.yaml` with hooks: ruff (--fix), black, isort, mypy
- [ ] Hooks run automatically on `git commit`
- [ ] `pyproject.toml` with black/isort/ruff/mypy configuration

### 14.3 Testing
- [ ] `pytest` + `pytest-asyncio` + `pytest-cov`
- [ ] `conftest.py`: fixtures for test docs, mock LLM, test FAISS store
- [ ] `test_ingestion.py`: loader, splitter, embedder, indexer unit tests
- [ ] `test_retrieval.py`: FAISS, BM25, hybrid, reranker, query expansion tests
- [ ] `test_chain.py`: LCEL chain end-to-end, memory, streaming tests
- [ ] `test_cache.py`: cache hit/miss, TTL tests
- [ ] `test_evaluation.py`: RAGAS metric computation on fixtures
- [ ] `tests/test_api/test_documents_api.py`: upload, list, delete endpoint tests
- [ ] `tests/test_api/test_query_api.py`: query, streaming endpoint tests
- [ ] `tests/test_api/test_evaluate_api.py`: evaluation endpoint tests
- [ ] Coverage target: ≥ 80% on `app/` and `rag/`
- [ ] `make test` runs all tests with coverage report

### 14.4 Makefile Shortcuts
- [ ] `make dev` — start backend + frontend in parallel (hot-reload)
- [ ] `make test` — pytest with coverage
- [ ] `make ingest` — ingest all docs in `data/uploads/`
- [ ] `make eval` — run RAGAS evaluation
- [ ] `make lint` — ruff + mypy
- [ ] `make format` — black + isort
- [ ] `make docker-up` — docker compose up --build -d
- [ ] `make docker-down` — docker compose down
- [ ] `make clean` — remove `__pycache__`, FAISS index, uploads, logs

---

## SECTION 15 · PROJECT STRUCTURE (Complete Folder Layout)

```
docmind/
│
├── .env                          # environment variables (git-ignored)
├── .env.example                  # all variables with safe defaults
├── .gitignore
├── .pre-commit-config.yaml
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── README.md                     # setup guide, demo GIF, RAGAS scores table
├── PRD.md
├── TECHSTACK.md
├── FEATURES_COMPLETE.md          # ← THIS FILE
├── requirements.txt              # pinned production deps
├── requirements-dev.txt          # dev/test deps
├── pyproject.toml                # black, isort, mypy, ruff config
│
├── app/                          # FastAPI backend
│   ├── main.py                   # app, lifespan, middleware, routers
│   ├── config.py                 # Pydantic Settings model
│   ├── dependencies.py           # FastAPI Depends() providers
│   ├── api/
│   │   ├── documents.py
│   │   ├── query.py
│   │   ├── conversations.py
│   │   ├── evaluate.py
│   │   ├── settings.py
│   │   └── health.py
│   ├── models/
│   │   ├── document.py
│   │   ├── query.py
│   │   ├── conversation.py
│   │   └── evaluation.py
│   └── middleware/
│       ├── logging.py
│       ├── timing.py
│       ├── api_key.py
│       └── rate_limiter.py       # NEW: per-session rate limiting
│
├── rag/                          # Core RAG pipeline
│   ├── ingestion/
│   │   ├── loader.py
│   │   ├── splitter.py
│   │   ├── embedder.py
│   │   ├── indexer.py
│   │   ├── ocr.py
│   │   ├── deduplicator.py       # NEW: hash-based duplicate detection
│   │   └── pipeline.py
│   ├── retrieval/
│   │   ├── faiss_retriever.py
│   │   ├── bm25_retriever.py
│   │   ├── hybrid.py             # EnsembleRetriever + RRF fusion
│   │   ├── reranker.py           # Cross-encoder reranking
│   │   ├── query_expansion.py    # MultiQueryRetriever, HyDE
│   │   ├── compressor.py         # LLMChainExtractor
│   │   ├── crag.py               # NEW: Corrective RAG relevance evaluator
│   │   ├── self_rag.py           # NEW: Self-RAG reflection loop
│   │   └── factory.py            # Builds full retriever from config
│   ├── generation/
│   │   ├── llm_factory.py
│   │   ├── chain.py
│   │   ├── memory.py
│   │   ├── streaming.py
│   │   └── cost_tracker.py
│   ├── cache/
│   │   └── semantic_cache.py
│   ├── evaluation/
│   │   ├── ragas_eval.py
│   │   ├── testset_gen.py
│   │   ├── drift.py
│   │   └── metrics.py            # NEW: Recall@k, NDCG, MRR computation
│   └── security/
│       ├── pii_redactor.py       # NEW: PII detection and redaction
│       └── access_control.py    # NEW: Document classification + role filtering
│
├── prompts/
│   ├── system_default.txt
│   ├── rag_qa.txt
│   ├── query_rewrite.txt
│   ├── hyde.txt
│   └── compression.txt
│
├── streamlit_app/
│   ├── app.py                    # Chat page
│   ├── utils.py
│   ├── components/
│   │   ├── chat_message.py
│   │   ├── source_expander.py
│   │   ├── retrieval_sidebar.py
│   │   └── metrics_card.py
│   └── pages/
│       ├── 1_Documents.py
│       ├── 2_Evaluation.py
│       ├── 3_Settings.py
│       └── 4_About.py
│
├── data/                         # runtime data (git-ignored)
│   ├── uploads/
│   ├── faiss_index/
│   │   ├── index.faiss
│   │   ├── index.pkl
│   │   └── bm25.pkl
│   └── metadata.json             # document registry
│
├── evaluation/
│   ├── testsets/
│   └── results/
│
├── logs/
│   └── app.log
│
└── tests/
    ├── conftest.py
    ├── test_ingestion.py
    ├── test_retrieval.py
    ├── test_chain.py
    ├── test_cache.py
    ├── test_evaluation.py
    └── test_api/
        ├── test_documents_api.py
        ├── test_query_api.py
        └── test_evaluate_api.py
```

---

## SECTION 16 · COMPLETE REQUIREMENTS.TXT

```
# Core RAG framework
langchain==0.2.16
langchain-community==0.2.16
langchain-openai==0.1.23
langchain-anthropic==0.1.23
langchain-ollama==0.1.3
langchain-huggingface==0.0.3

# API
fastapi==0.111.0
uvicorn[standard]==0.29.0
python-multipart==0.0.9
sse-starlette==2.1.0
pydantic==2.7.4
pydantic-settings==2.3.4

# Frontend
streamlit==1.35.0
streamlit-extras==0.4.3

# Vector store + retrieval
faiss-cpu==1.8.0
rank-bm25==0.2.2

# Embeddings + reranking
sentence-transformers==3.1.0
torch==2.3.0

# LLM clients
openai==1.35.13
anthropic==0.29.0

# Document processing
pypdf==4.2.0
unstructured[pdf]==0.14.0
pytesseract==0.3.10
pdf2image==1.17.0
Pillow==10.3.0
python-docx==1.1.2
docx2txt==0.8
chardet==5.2.0

# Evaluation
ragas==0.1.21
datasets==2.20.0

# Cache
redis==5.0.7
langchain-redis==0.0.2

# Observability
prometheus-client==0.20.0
python-json-logger==2.0.7
evidently==0.4.33

# Security
presidio-analyzer==2.2.354       # PII detection
presidio-anonymizer==2.2.354     # PII redaction
slowapi==0.1.9                   # Rate limiting for FastAPI

# Utilities
python-dotenv==1.0.1
httpx==0.27.0
tenacity==8.4.1                  # Retry with exponential backoff
numpy==1.26.4
pandas==2.2.2
hashlib                          # stdlib - for file deduplication
```

### Dev / Test Requirements (`requirements-dev.txt`)
```
pytest==8.2.2
pytest-asyncio==0.23.7
pytest-cov==5.0.0
httpx==0.27.0
black==24.4.2
isort==5.13.2
ruff==0.4.9
mypy==1.10.1
pre-commit==3.7.1
```

---

## SECTION 17 · FEATURE DECISION MATRIX

This matrix confirms every feature is implemented and explains its purpose:

| Feature | Category | Why It's Here |
|---------|----------|---------------|
| Hybrid FAISS + BM25 + RRF | Retrieval | Outperforms either alone; industry standard |
| MMR retrieval | Retrieval | Balances relevance + diversity |
| Cross-encoder reranking | Post-retrieval | Precision boost; reduces hallucination |
| Query rewriting | Pre-retrieval | Handles vague/short queries |
| Multi-query expansion | Pre-retrieval | Semantic diversity, higher recall |
| HyDE | Pre-retrieval | Best for abstract/conceptual questions |
| CRAG relevance check | Post-retrieval | Catches low-quality retrievals before LLM |
| Self-RAG reflection | Generation | Self-corrects ungrounded answers |
| Context compression | Post-retrieval | Reduces tokens, improves signal |
| Parent document retrieval | Post-retrieval | Richer context for reranked child chunks |
| Diversity deduplication | Post-retrieval | Prevents near-duplicate chunks in context |
| Chunk metadata stamps | Ingestion | Enables exact page citations |
| OCR fallback | Ingestion | Handles scanned PDFs |
| Table-aware chunking | Ingestion | Preserves tabular data meaning |
| SemanticChunker | Ingestion | Better splits for unstructured text |
| ParentDocumentRetriever | Ingestion | Dual-granularity indexing |
| RAGAS evaluation | Evaluation | Objective RAG quality metrics |
| Synthetic test set gen | Evaluation | Benchmarking without manual labelling |
| Grounding score / USR | Evaluation | Measures hallucination rate |
| Recall@k / NDCG / MRR | Evaluation | Retrieval quality measurement |
| Drift detection | Evaluation | Catches degradation in production |
| Streaming SSE | Generation | No waiting; live token output |
| Semantic caching | Performance | Repeat queries instant, saves API cost |
| Redis cache | Performance | Production-grade persistent cache |
| Confidence badge | UI | Trust signal for legal/medical users |
| Source citation expanders | UI | Traceable, verifiable answers |
| Thumbs up/down feedback | UI | Continuous improvement signal |
| Conversation memory | Generation | Natural multi-turn Q&A |
| Session cost display | UI | Transparency on API spend |
| Export chat PDF/MD | UI | Archive and share Q&A sessions |
| Document tags + filtering | Document Mgmt | Scoped retrieval to relevant subset |
| Chunk preview modal | Document Mgmt | Verify chunking quality visually |
| Structured JSON logging | Observability | Machine-readable logs for monitoring |
| Prometheus metrics | Observability | Grafana-compatible scraping |
| LangSmith tracing | Observability | Debug per-step latency breakdown |
| Audit trail (tamper-evident) | Security | HIPAA/legal compliance |
| PII redaction | Security | Protect sensitive output |
| Rate limiting | Security | Prevent abuse |
| Document classification | Security | Access control for sensitive docs |
| Input sanitisation | Security | Block prompt injection |
| API key auth | Security | Protect REST API |
| Type hints + mypy | Code Quality | Catch bugs at development time |
| Pre-commit hooks | Code Quality | Enforce quality on every commit |
| Full test suite | Code Quality | Prevent regressions |
| Docker Compose | Deployment | One-command full-stack deploy |
| Health check endpoint | Deployment | Docker liveness probe |
| `.env.example` | Config | Reproducible setup |
| Pydantic Settings | Config | Type-safe config with validation |
| Makefile | DX | Shortcut commands for all workflows |
| Versioned prompt templates | Prompts | Reproducible, tunable prompts |
| Fallback chain (LLM + embed) | Reliability | No downtime if primary provider fails |
| Idempotent ingestion | Reliability | Safe to retry on failure |
| Exponential backoff | Reliability | Handles transient API failures |

---

*This is the definitive and exhaustive feature list for DocMind. Every item above must be implemented. No additional features are required beyond this document.*
