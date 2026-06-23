# Product Requirements Document
## DocMind — AI-Powered RAG Document Q&A System

**Version:** 1.0.0  
**Status:** Ready for Development  
**Stack Alignment:** LangChain · FAISS · FastAPI · Streamlit · OpenAI / HuggingFace  

---

## 1. Product Overview

DocMind is a production-grade, full-stack Retrieval-Augmented Generation (RAG) application that enables users to upload documents and query them in natural language. The system retrieves semantically relevant chunks using a hybrid search strategy (dense embeddings + BM25), reranks them with a cross-encoder, and generates grounded, cited answers using an LLM. Every answer references its source page and chunk, making hallucination traceable and eliminating it structurally.

**Target users:** Students, researchers, legal/medical professionals, enterprise knowledge-base teams.  
**Primary differentiator:** Every advertised production RAG technique—hybrid search, MMR, reranking, RAGAS evaluation, query expansion, caching, streaming, drift detection—is built-in, not bolted on.

---

## 2. Goals & Non-Goals

### Goals
- Upload and index any document (PDF, DOCX, TXT, MD, CSV, HTML)
- Conversational Q&A with full chat history over indexed documents
- Cited, grounded answers with exact page/chunk references
- Hybrid retrieval (dense + BM25 + RRF fusion)
- Post-retrieval cross-encoder reranking
- Query expansion and HyDE (Hypothetical Document Embeddings)
- Streaming LLM responses with live source highlighting
- RAGAS-based automated evaluation dashboard
- Semantic response caching for repeat queries
- Full MLOps: logging, tracing, cost tracking, drift detection
- Docker-first, one-command deploy

### Non-Goals
- Real-time web crawling or live internet search (scoped to uploaded docs)
- Multi-tenant SaaS billing (can extend later)
- Mobile native app (Streamlit is responsive enough for MVP)

---

## 3. User Stories

### 3.1 Document Management
| ID | As a... | I want to... | So that... |
|----|---------|-------------|------------|
| U01 | User | Upload one or multiple documents at once | I can build a knowledge base across files |
| U02 | User | See an upload progress bar with per-file status | I know ingestion is working |
| U03 | User | View all indexed documents with metadata (name, pages, size, date, chunk count) | I can manage my knowledge base |
| U04 | User | Delete individual documents and re-trigger re-indexing | I can keep the knowledge base clean |
| U05 | User | See chunk previews for any indexed document | I can verify chunking quality |
| U06 | User | Download the indexed document back | I can retrieve originals |
| U07 | User | Tag documents with labels (e.g. "legal", "research") | I can filter Q&A scope to a subset |
| U08 | Admin | View storage usage and document count metrics | I can monitor system health |

### 3.2 Querying & Chat
| ID | As a... | I want to... | So that... |
|----|---------|-------------|------------|
| Q01 | User | Type a natural language question and get a cited answer | I can extract information without reading documents |
| Q02 | User | See the exact source chunks used in each answer, with page numbers | I can verify and trust the answer |
| Q03 | User | Click on a source citation to see the full chunk highlighted | I can read surrounding context |
| Q04 | User | Have multi-turn conversations with memory of prior turns | I can ask follow-up questions naturally |
| Q05 | User | Clear the conversation history and start fresh | I can reset context when needed |
| Q06 | User | See the LLM's answer stream in real-time (token by token) | I don't wait for a full response |
| Q07 | User | Copy any answer to clipboard with one click | I can use answers in other tools |
| Q08 | User | Export a full conversation as PDF or Markdown | I can share or archive Q&A sessions |
| Q09 | User | Rate each answer thumbs up/down with optional comment | I can give feedback to improve the system |
| Q10 | User | See a confidence score or relevance badge on each answer | I know how reliable the retrieval was |
| Q11 | User | Filter Q&A scope to specific documents or tags | I can ask targeted questions |
| Q12 | User | Ask follow-up questions referencing prior answers ("What did you mean by...") | Conversations feel natural |
| Q13 | User | Regenerate any answer with a different retrieval strategy | I can compare output quality |

### 3.3 Search & Retrieval
| ID | As a... | I want to... | So that... |
|----|---------|-------------|------------|
| S01 | User | Get answers that blend keyword and semantic matches | Exact terms AND meaning are both found |
| S02 | User | Choose between similarity-based and MMR retrieval | I control diversity vs precision of results |
| S03 | System | Expand vague queries automatically using the LLM | Short or ambiguous queries still retrieve well |
| S04 | System | Rerank retrieved chunks using a cross-encoder | The most relevant chunks reach the LLM first |
| S05 | System | Cache semantically similar queries and return cached responses | Repeat questions are instant |
| S06 | User | Adjust top-k, chunk overlap, and retrieval mode from the sidebar | I can tune retrieval without code |

### 3.4 Evaluation & Observability
| ID | As a... | I want to... | So that... |
|----|---------|-------------|------------|
| E01 | User | See a RAGAS scorecard (faithfulness, answer relevance, context precision, context recall) | I know my RAG pipeline quality objectively |
| E02 | Admin | Run a synthetic Q&A evaluation batch against indexed documents | I can benchmark the system before deployment |
| E03 | Admin | View query logs, latency histograms, and token usage over time | I can monitor and optimize |
| E04 | Admin | Get an alert when retrieval quality drops below a threshold | Drift is caught automatically |
| E05 | Developer | See full LangSmith-compatible traces for each query | I can debug retrieval + generation steps |
| E06 | Admin | Export all evaluation metrics as CSV | I can include them in reports or READMEs |

### 3.5 Configuration & Settings
| ID | As a... | I want to... | So that... |
|----|---------|-------------|------------|
| C01 | User | Switch LLM provider (OpenAI GPT-4o, GPT-4o-mini, Claude, local Ollama) from UI | I can trade off cost vs quality |
| C02 | User | Switch embedding model (OpenAI, HuggingFace sentence-transformers) | I control indexing cost and quality |
| C03 | User | Set system prompt / persona for the assistant | I can customize behaviour for my use case |
| C04 | User | Control context window size and max tokens | I can optimise for cost |
| C05 | Admin | Set rate limits per session | I can prevent abuse |
| C06 | User | Enable/disable query expansion, reranking, caching from sidebar toggles | I can AB test strategies live |

---

## 4. Functional Requirements

### 4.1 Document Ingestion Pipeline

#### 4.1.1 Supported File Types
- PDF (text-layer + scanned via OCR fallback using `pytesseract`)
- DOCX (via `python-docx`)
- TXT, Markdown (`.md`)
- CSV (table-aware chunking — each row = one chunk)
- HTML (tag-stripped)
- Images in PDFs are captioned via LLM vision (optional, toggle)

#### 4.1.2 Preprocessing
- Encoding detection (`chardet`) and normalisation to UTF-8
- Whitespace collapse, ligature normalisation, header/footer stripping
- Table detection and preservation as structured text
- Metadata extraction: title, author, page count, word count, creation date

#### 4.1.3 Chunking Strategy
- Default: `RecursiveCharacterTextSplitter` (chunk_size=1000, overlap=200)
- Alternative: `SemanticChunker` (groups sentences by embedding distance)
- Alternative: `ParentDocumentRetriever` pattern (small chunks for retrieval, large for context)
- Chunk metadata stamps: source file, page number, chunk index, char offset
- Configurable via settings panel

#### 4.1.4 Embedding
- Primary: `text-embedding-3-small` (OpenAI) — best cost/quality ratio
- Fallback: `all-MiniLM-L6-v2` (HuggingFace, fully local, no API key required)
- Fallback 2: `BAAI/bge-large-en-v1.5` (HuggingFace, higher accuracy)
- Batch embedding with retry on rate-limit errors (exponential backoff)

#### 4.1.5 Indexing
- Primary store: FAISS (`IndexFlatL2` for ≤100k chunks, `IndexIVFFlat` for larger)
- FAISS index persisted to disk at `data/faiss_index/`
- BM25 sparse index maintained in parallel (`rank_bm25`)
- Index rebuild triggered on document add/delete
- Async ingestion — UI shows progress, API returns job ID for polling

### 4.2 Retrieval Pipeline

#### 4.2.1 Query Processing (Pre-Retrieval)
1. **Query rewriting** — LLM rewrites vague or short queries into explicit search-friendly form
2. **Multi-query expansion** — LLM generates 3 semantically diverse versions of the query; results are union-merged with deduplication
3. **HyDE (Hypothetical Document Embeddings)** — LLM generates a hypothetical answer; that answer is embedded and used as query vector (improves recall for abstract questions)
4. **Conversation history fusion** — last N turns are summarised and injected into query context

#### 4.2.2 Retrieval
1. Dense retrieval: FAISS cosine similarity, top-20 candidates
2. Sparse retrieval: BM25 on raw text, top-20 candidates
3. RRF fusion (Reciprocal Rank Fusion): merge and re-score both lists into unified top-30
4. Retrieval modes selectable per query: Similarity | MMR | Hybrid (default)
5. Metadata filtering: filter by document name, tag, or page range before retrieval

#### 4.2.3 Post-Retrieval Processing
1. **Cross-encoder reranking** — `cross-encoder/ms-marco-MiniLM-L-6-v2` reranks top-30 to top-5
2. **Context compression** — `LLMChainExtractor` removes irrelevant sentences from each chunk (reduces token count, improves signal)
3. **Parent document expansion** — reranked child chunks retrieve their parent chunk for richer context
4. **Diversity check** — deduplicate nearly identical chunks (>0.95 cosine sim between retrieved chunks)

### 4.3 Generation Pipeline

#### 4.3.1 Prompt Design
- System prompt includes: persona, grounding instruction, citation format, fallback phrase ("I don't have enough information")
- Context formatted as: `[Source: filename, Page: N] chunk text`
- Chat history injected as alternating user/assistant turns
- Prompt templates versioned in `prompts/` directory

#### 4.3.2 LLM Integration
- Primary: OpenAI `gpt-4o-mini` (default, cheapest with strong accuracy)
- Option: `gpt-4o` (higher accuracy, higher cost)
- Option: Claude via Anthropic API
- Option: Local Ollama (`llama3`, `mistral`) — zero API cost, runs locally
- LangChain LCEL chain: retriever → context formatter → prompt → LLM → output parser

#### 4.3.3 Output
- Streaming response via SSE (Server-Sent Events)
- Final answer + list of source chunks (file, page, excerpt)
- Confidence estimation (average reranker score of top-k chunks)
- Token usage logged per query

### 4.4 Semantic Cache
- Engine: `InMemorySemanticCache` (via LangChain) with cosine similarity threshold (default 0.95)
- Persistent cache: `redis-py` + `langchain-redis` for production
- Cache TTL: configurable (default 24h)
- Cache hit/miss logged for observability
- Cache can be cleared from admin panel

### 4.5 Evaluation Module
- RAGAS metrics computed on demand or on schedule:
  - **Faithfulness** — is the answer factually consistent with the retrieved context?
  - **Answer Relevance** — does the answer address the question?
  - **Context Precision** — are the retrieved chunks relevant?
  - **Context Recall** — does the context cover the ground-truth answer?
- Synthetic test set generation using RAGAS `TestsetGenerator`
- Results stored in `evaluation/results/` as CSV + JSON
- Visual dashboard in Streamlit: metric trends over time (line chart), per-query breakdown (table)

### 4.6 Observability & Logging
- Structured logging (JSON) via Python `logging` + `python-json-logger`
- Per-query trace: query text, rewritten query, retrieved chunks, reranker scores, LLM tokens, latency, cache hit
- Cost tracking: token count × model price per query and cumulative total
- LangSmith integration (optional): full LCEL chain traces via `LANGSMITH_API_KEY`
- Prometheus metrics exposed at `/metrics` (query count, latency p50/p95, error rate)
- Drift detection: Evidently AI profile on embedding distribution — alert if input drift exceeds threshold

### 4.7 REST API (FastAPI)
All Streamlit UI functionality exposed as a versioned REST API for integration:

```
POST   /api/v1/documents/upload        Upload and ingest documents
GET    /api/v1/documents               List all indexed documents
DELETE /api/v1/documents/{id}          Delete document and re-index
GET    /api/v1/documents/{id}/chunks   List chunks for a document

POST   /api/v1/query                   Single Q&A query (JSON response)
POST   /api/v1/query/stream            Streaming Q&A (SSE)
GET    /api/v1/conversations           List conversation sessions
POST   /api/v1/conversations           Create new conversation session
GET    /api/v1/conversations/{id}      Get conversation history
DELETE /api/v1/conversations/{id}      Delete conversation

POST   /api/v1/evaluate               Run RAGAS evaluation on current index
GET    /api/v1/evaluate/results        Get evaluation scores
GET    /api/v1/evaluate/export         Download scores as CSV

GET    /api/v1/health                  Health check
GET    /metrics                        Prometheus metrics
GET    /api/v1/settings                Get current config
PATCH  /api/v1/settings                Update config (LLM, retrieval mode, etc.)
```

- FastAPI with automatic OpenAPI docs at `/docs`
- Pydantic v2 request/response models
- API key authentication via `X-API-Key` header (configurable, can be disabled for local dev)
- CORS configured for Streamlit frontend origin

### 4.8 Settings & Configuration
All settings manageable from UI sidebar AND `.env` file:

| Setting | Default | Options |
|---------|---------|---------|
| LLM provider | `openai` | `openai`, `anthropic`, `ollama` |
| LLM model | `gpt-4o-mini` | any provider model name |
| Embedding model | `text-embedding-3-small` | see §4.1.4 |
| Chunk size | `1000` | 200–4000 |
| Chunk overlap | `200` | 0–1000 |
| Top-k retrieval | `20` | 1–50 |
| Reranker top-k | `5` | 1–20 |
| Retrieval mode | `hybrid` | `similarity`, `mmr`, `hybrid` |
| Query expansion | `true` | bool |
| HyDE | `false` | bool |
| Reranking | `true` | bool |
| Context compression | `false` | bool |
| Cache enabled | `true` | bool |
| Cache TTL | `86400` | seconds |
| Temperature | `0.0` | 0.0–1.0 |
| Max tokens | `1024` | 128–8192 |
| System prompt | (default) | free text |
| API key auth | `false` | bool |

---

## 5. Non-Functional Requirements

### 5.1 Performance
- Query response time: < 3s p50, < 8s p95 (excluding streaming tail)
- Document ingestion: ≥ 10 pages/second
- FAISS index search: < 50ms for 100k vectors
- Cache hit response: < 200ms

### 5.2 Reliability
- Graceful degradation: if reranker or LLM fails, fall back to raw retrieval + simpler model
- Retry with exponential backoff on all external API calls (openai, anthropic)
- All ingestion jobs idempotent — safe to retry
- Ingestion errors surface clearly in UI with human-readable messages

### 5.3 Security
- API keys stored in `.env` only, never in code or logs
- Uploaded documents stored in `data/uploads/`, not served publicly
- Optional API key header auth for REST API
- Input sanitisation on all user text fields
- No PII logged (query text logging can be disabled)

### 5.4 Maintainability
- 100% type-annotated Python (mypy clean)
- Docstrings on all public functions
- Black + isort + ruff for formatting/linting
- Pre-commit hooks for all the above
- Unit tests for ingestion, retrieval, and chain logic (pytest)
- Integration tests for API endpoints

### 5.5 Portability
- Docker Compose: single `docker-compose up` to run frontend + backend + Redis
- `.env.example` with all required and optional variables documented
- No hard-coded paths — all configurable via environment

---

## 6. UI/UX Requirements

### 6.1 Streamlit Pages
The app is structured as a multi-page Streamlit app:

| Page | Route | Purpose |
|------|-------|---------|
| Chat | `app.py` (default) | Main Q&A interface |
| Documents | `pages/1_Documents.py` | Upload, manage, preview |
| Evaluation | `pages/2_Evaluation.py` | RAGAS dashboard |
| Settings | `pages/3_Settings.py` | Config, model selector |
| About | `pages/4_About.py` | Tech stack, README |

### 6.2 Chat Page Layout
```
┌─────────────────────────────────────────────────────────────┐
│  Sidebar                    │  Main Chat Area               │
│  ─────────────────          │  ────────────────────────     │
│  Document scope filter      │  Message history              │
│  Retrieval mode selector    │  [User] What is...?           │
│  Top-k slider               │  [Bot] According to [p.3]...  │
│  Temperature slider         │       ▶ Source 1 (expand)    │
│  Toggle: Query expansion    │       ▶ Source 2 (expand)    │
│  Toggle: Reranking          │  ─────────────────────────   │
│  Toggle: Cache              │  [input box]    [Send] [Clear]│
│  ─────────────────          │                               │
│  Conversation history       │  Cost this session: $0.003   │
│  ─────────────────          │  Tokens: 1,204               │
│  Export chat (PDF/MD)       │                               │
└─────────────────────────────────────────────────────────────┘
```

### 6.3 Source Citation UI
Each answer renders inline citations:
- Collapsible expander per source: `[Page 3 · documents/report.pdf]`
- Shows: page number, chunk text (highlighted), relevance score badge
- Confidence badge on answer: `High ●`, `Medium ●`, `Low ●` based on reranker score

### 6.4 Document Page
- Drag-and-drop multi-file uploader with progress per file
- Table of indexed docs: name, type, pages, chunks, indexed date, tags, actions (delete, preview)
- Chunk preview modal: shows all chunks for selected document with metadata
- Re-index button (per document or all)

### 6.5 Evaluation Page
- Run evaluation button (triggers RAGAS on current index using synthetic test set)
- Metric cards: Faithfulness, Answer Relevance, Context Precision, Context Recall
- Line chart: metric scores over evaluation runs
- Per-query breakdown table (downloadable)
- Export to CSV button

---

## 7. Data Model

### 7.1 Document Record
```python
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
  "status": "indexed"  # queued | processing | indexed | failed
}
```

### 7.2 Query Log
```python
{
  "query_id": "uuid",
  "session_id": "uuid",
  "original_query": "What is the revenue?",
  "rewritten_query": "What was the total revenue figure reported in the document?",
  "retrieval_mode": "hybrid",
  "chunks_retrieved": 20,
  "chunks_after_rerank": 5,
  "top_chunk_score": 0.91,
  "llm_model": "gpt-4o-mini",
  "prompt_tokens": 1450,
  "completion_tokens": 280,
  "cost_usd": 0.00028,
  "latency_ms": 2340,
  "cache_hit": false,
  "timestamp": "2025-06-20T10:35:00Z"
}
```

### 7.3 Evaluation Result
```python
{
  "run_id": "uuid",
  "timestamp": "2025-06-20T11:00:00Z",
  "question_count": 50,
  "faithfulness": 0.87,
  "answer_relevancy": 0.91,
  "context_precision": 0.84,
  "context_recall": 0.79,
  "per_question": [...]
}
```

---

## 8. Error States & Edge Cases

| Scenario | Behaviour |
|----------|-----------|
| Document has no extractable text (scanned) | OCR fallback triggered; user notified |
| LLM API rate limit | Exponential backoff, max 3 retries; error shown if all fail |
| Query retrieves 0 chunks | "No relevant content found" response, no LLM call |
| LLM returns answer not in context | Low confidence badge; answer still shown |
| FAISS index file missing/corrupt | Auto-rebuild from stored chunk metadata |
| Upload of unsupported file type | Clear error message with supported types listed |
| Context window exceeded | Chunks truncated from tail; user notified via warning banner |

---

## 9. Evaluation Criteria (Resume / Hackathon)

This project is specifically designed to showcase:

| Criterion | Implemented Feature |
|-----------|-------------------|
| Production ML, not just notebooks | FastAPI backend, Docker, structured logging |
| Advanced retrieval | Hybrid search, MMR, HyDE, reranking |
| Responsible AI | Source citation, confidence scores, grounding guardrail |
| Evaluation culture | RAGAS scorecard, per-metric trend charts |
| MLOps awareness | Cost tracking, drift detection, Prometheus metrics |
| Full-stack | Streamlit UI + FastAPI REST API + Docker Compose |
| Code quality | Type hints, tests, linting, pre-commit |
| Documentation | This PRD, TECHSTACK.md, OpenAPI docs at `/docs` |

---

## 10. Out of Scope (Future Roadmap)

- GraphRAG / knowledge graph integration
- Multi-user auth (OAuth2 / JWT)
- Fine-tuning embedding models on domain data
- Voice input / TTS output
- Slack / Teams bot integration
- Multi-language document support
- Real-time document sync (Google Drive, Notion, Confluence connectors)
- Agentic multi-hop reasoning (LangGraph)
