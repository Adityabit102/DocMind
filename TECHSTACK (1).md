# Tech Stack
## DocMind — RAG Document Q&A System

**Last updated:** June 2025  
**Python:** 3.11+  
**Compatibility guarantee:** All packages listed are mutually compatible at the pinned versions below.

---

## Stack Summary

| Layer | Technology | Version | Role |
|-------|-----------|---------|------|
| Frontend | Streamlit | 1.35.0 | Multi-page chat + document management UI |
| API | FastAPI | 0.111.0 | REST API, SSE streaming, OpenAPI docs |
| RAG Framework | LangChain | 0.2.x (LCEL) | Pipeline orchestration, chain composition |
| LangChain Community | langchain-community | 0.2.x | Loaders, vector store wrappers, BM25 |
| LangChain OpenAI | langchain-openai | 0.1.x | OpenAI LLM + embedding integration |
| Vector Store | FAISS (CPU) | 1.8.0 | Dense vector indexing and similarity search |
| Sparse Retrieval | rank-bm25 | 0.2.2 | BM25 keyword index |
| Reranker | sentence-transformers | 3.1.0 | Cross-encoder reranking |
| Embeddings (local) | sentence-transformers | 3.1.0 | HuggingFace embedding models |
| LLM (primary) | OpenAI API | — | GPT-4o-mini / GPT-4o generation |
| LLM (local fallback) | Ollama | latest | Local LLM inference (llama3, mistral) |
| LLM (alternative) | Anthropic API | — | Claude models |
| Evaluation | RAGAS | 0.1.x | Faithfulness, relevance, precision, recall |
| Cache | InMemorySemanticCache | LangChain built-in | Local dev semantic caching |
| Cache (prod) | Redis | 7.2 | Production-grade persistent cache |
| OCR | pytesseract + Pillow | latest | Scanned PDF text extraction |
| Observability | LangSmith | — | Full LCEL chain tracing (optional) |
| Metrics | prometheus-client | 0.20.0 | Prometheus `/metrics` endpoint |
| Drift Detection | evidently | 0.4.x | Embedding distribution drift alerts |
| Data Validation | Pydantic v2 | 2.7.0 | API request/response models |
| HTTP Server | Uvicorn | 0.29.0 | ASGI server for FastAPI |
| Logging | python-json-logger | 2.0.7 | Structured JSON logs |
| Env Management | python-dotenv | 1.0.1 | `.env` file loading |
| Containerisation | Docker + Compose | latest | One-command deployment |
| Linting | Ruff | 0.4.x | Fast Python linter |
| Formatting | Black | 24.x | Python code formatter |
| Import sorting | isort | 5.13.x | Import organisation |
| Testing | pytest + pytest-asyncio | latest | Unit + integration tests |
| Type checking | mypy | 1.10.x | Static type analysis |
| Pre-commit | pre-commit | 3.7.x | Git hooks for lint/format |

---

## 1. Frontend — Streamlit 1.35

**Why Streamlit over Flask/Gradio/Next.js:**  
Streamlit renders a full multi-page app with file upload, chat, charts, and data tables in pure Python. Zero JavaScript needed — perfect for a solo fresher portfolio project where backend depth matters more than frontend polish. It's also the standard for AI/ML demo apps in industry.

**Key Streamlit features used:**
- `st.chat_message` / `st.chat_input` — native chat UI components
- `st.write_stream` — streams LLM tokens directly into the UI
- `st.session_state` — persists conversation history across reruns
- `st.file_uploader(accept_multiple_files=True)` — multi-document upload
- `st.sidebar` — retrieval control panel (sliders, toggles, selectors)
- `st.expander` — collapsible source citation blocks
- `st.progress` + `st.status` — ingestion progress feedback
- `st.metric` — RAGAS score cards
- `st.line_chart` — metric trend over evaluation runs
- `st.data_editor` — editable document metadata table
- `st.tabs` — organise chat + sources side by side
- `st.toast` — non-blocking success/error notifications
- `st.cache_resource` — cache the FAISS index and LLM client across sessions
- Multi-page structure via `pages/` directory

```
requirements (frontend only):
streamlit==1.35.0
streamlit-extras==0.4.3      # copy-to-clipboard, metric cards, etc.
```

---

## 2. API — FastAPI 0.111 + Uvicorn

**Why FastAPI:**  
Industry-standard Python async web framework. Auto-generates OpenAPI docs. Native Pydantic v2 integration for validation. Supports SSE (Server-Sent Events) natively for streaming. Faster than Flask for async workloads.

**Key features used:**
- `APIRouter` — modular route grouping (`/documents`, `/query`, `/evaluate`)
- `StreamingResponse` with `media_type="text/event-stream"` — SSE streaming
- `BackgroundTasks` — async document ingestion jobs
- `Depends()` — dependency injection for shared resources (FAISS store, LLM client)
- `HTTPException` with custom error models
- Middleware: CORS, request logging, response timing
- Lifespan events: load FAISS index on startup, flush cache on shutdown
- `/docs` — Swagger UI (auto-generated)
- `/redoc` — ReDoc alternative docs
- `/metrics` — Prometheus metrics via `prometheus-client`
- `/api/v1/health` — liveness probe for Docker health checks

```
requirements (API):
fastapi==0.111.0
uvicorn[standard]==0.29.0
python-multipart==0.0.9   # file upload support
sse-starlette==2.1.0      # SSE streaming helper
```

---

## 3. RAG Framework — LangChain (LCEL)

**Why LangChain over LlamaIndex / bare code:**  
LangChain 0.2 with LCEL (LangChain Expression Language) is the industry reference for building RAG pipelines. LCEL provides declarative, composable chains with built-in streaming, parallelism, retries, and fallbacks. Recruiters and interviewers recognise it immediately.

**Core LangChain components used:**

### Document Loading
```python
# All loaders from langchain_community.document_loaders
PyPDFLoader          # PDF (text layer)
UnstructuredPDFLoader # PDF (OCR fallback, table extraction)
Docx2txtLoader       # DOCX
TextLoader           # TXT, MD
CSVLoader            # CSV (row-per-chunk)
UnstructuredHTMLLoader # HTML
```

### Text Splitting
```python
# Primary
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,    # preserves char offset for citation
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Alternative (switchable from UI)
SemanticChunker(embeddings)  # splits at embedding distance breakpoints
```

### Retrieval Architecture
```python
# FAISS dense retriever
FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 20, "fetch_k": 40, "lambda_mult": 0.5}
)

# BM25 sparse retriever
BM25Retriever.from_documents(chunks, k=20)

# Hybrid fusion (Ensemble)
EnsembleRetriever(
    retrievers=[faiss_retriever, bm25_retriever],
    weights=[0.6, 0.4]      # tunable via settings
)

# Query expansion (Multi-query)
MultiQueryRetriever.from_llm(retriever=ensemble, llm=llm)

# Context compression (post-retrieval)
LLMChainExtractor.from_llm(llm)
ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

# Parent document retrieval
ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=InMemoryStore(),
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)
```

### Chain Composition (LCEL)
```python
rag_chain = (
    {
        "context": retriever | format_docs_with_citations,
        "question": RunnablePassthrough(),
        "chat_history": RunnableLambda(get_history)
    }
    | prompt_template
    | llm
    | StrOutputParser()
)
# Streaming:
rag_chain.stream({"question": query})
# Async:
await rag_chain.ainvoke({"question": query})
```

### Memory & History
```python
ConversationBufferWindowMemory(k=5)   # last 5 turns
ConversationSummaryMemory(llm=llm)    # summarises older turns
# Injected into chain via MessagesPlaceholder
```

### Semantic Cache
```python
set_llm_cache(InMemorySemanticCache(
    embedding=embeddings,
    score_threshold=0.95
))
# Production:
set_llm_cache(RedisSemanticCache(
    embedding=embeddings,
    redis_url="redis://redis:6379"
))
```

```
requirements (LangChain):
langchain==0.2.16
langchain-community==0.2.16
langchain-openai==0.1.23
langchain-anthropic==0.1.23    # Claude support
langchain-ollama==0.1.3        # Ollama local support
langchain-huggingface==0.0.3   # HuggingFace embeddings
```

---

## 4. Vector Store — FAISS

**Why FAISS over Pinecone/Chroma/Qdrant:**  
FAISS is local, free, and ultra-fast. No external service needed. Perfect for a portfolio project and interviews. It can handle millions of vectors on a CPU. LangChain's FAISS wrapper adds `save_local` / `load_local` for persistence.

**Index selection logic:**
```python
if n_chunks < 100_000:
    index = faiss.IndexFlatL2(dim)           # exact, no config needed
else:
    index = faiss.IndexIVFFlat(              # approximate, faster at scale
        faiss.IndexFlatL2(dim), dim,
        nlist=int(sqrt(n_chunks))
    )
```

**Persistence:**
```python
vectorstore.save_local("data/faiss_index")
vectorstore = FAISS.load_local(
    "data/faiss_index", embeddings,
    allow_dangerous_deserialization=True
)
```

**FAISS + BM25 together implement full hybrid RAG without any cloud service.**

```
requirements:
faiss-cpu==1.8.0
rank-bm25==0.2.2
```

---

## 5. Reranker — sentence-transformers Cross-Encoder

**Why reranking:**  
FAISS retrieves by embedding similarity (bi-encoder), which is fast but imprecise. A cross-encoder reads query AND chunk together and outputs a precise relevance score. Reranking top-20 FAISS results to top-5 meaningfully reduces hallucination.

**Model used:** `cross-encoder/ms-marco-MiniLM-L-6-v2`  
- Small (22M params), fast on CPU (< 100ms for 20 candidates)
- Trained on MS MARCO — strong out-of-the-box relevance for English documents

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = reranker.predict([(query, chunk.page_content) for chunk in candidates])
ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
top_k = [doc for doc, _ in ranked[:5]]
```

**HyDE (Hypothetical Document Embeddings):**
```python
# Generate a hypothetical answer, embed it, use as query vector
hyde_prompt = "Write a passage that answers: {question}"
hypothetical_doc = llm.invoke(hyde_prompt.format(question=query))
hyde_embedding = embeddings.embed_query(hypothetical_doc)
```

```
requirements:
sentence-transformers==3.1.0
torch==2.3.0            # CPU-only; GPU automatically if available
```

---

## 6. Embeddings

### Primary: OpenAI `text-embedding-3-small`
- 1536-dimensional dense vectors
- Best cost/quality ratio in 2025: $0.02/1M tokens
- Requires `OPENAI_API_KEY`

### Local Fallback: `all-MiniLM-L6-v2`
- 384-dimensional, runs on CPU, no API key
- Sufficient for most document Q&A use cases
- Activated automatically if `OPENAI_API_KEY` not set

### High-accuracy option: `BAAI/bge-large-en-v1.5`
- 1024-dimensional, best open-source English embedding
- Slower than MiniLM but significantly more accurate
- Toggle in settings

---

## 7. LLM Providers

### OpenAI (Primary)
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
```
Models available: `gpt-4o-mini` (default), `gpt-4o`, `gpt-3.5-turbo`

### Anthropic Claude (Alternative)
```python
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
```
Models: `claude-3-5-sonnet-20241022`, `claude-3-haiku-20240307`

### Ollama (Local, Zero Cost)
```python
from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3", base_url="http://ollama:11434")
```
Models: `llama3`, `mistral`, `phi3`, `gemma2`  
Pull command: `ollama pull llama3`

**Provider selection:** Configured via `LLM_PROVIDER` env var or UI dropdown. Falls back in order: OpenAI → Anthropic → Ollama.

---

## 8. Document Processing

### PDF Text Extraction
```python
PyPDFLoader                  # fast, text-only, page metadata
UnstructuredPDFLoader        # tables, headers, multi-column layouts
```

### OCR Fallback (Scanned PDFs)
```python
pytesseract                  # Tesseract OCR wrapper
pdf2image                    # converts PDF pages to PIL Images
Pillow                       # image preprocessing
```
Triggered automatically when `PyPDFLoader` extracts < 50 chars/page.

### DOCX
```python
Docx2txtLoader               # simple text extraction
python-docx                  # structured access to paragraphs, tables, styles
```

### Encoding Detection
```python
chardet                      # detects encoding of text files
```

```
requirements (document processing):
pypdf==4.2.0
unstructured[pdf]==0.14.0
pytesseract==0.3.10
pdf2image==1.17.0
Pillow==10.3.0
python-docx==1.1.2
docx2txt==0.8
chardet==5.2.0
```

---

## 9. Evaluation — RAGAS

**What RAGAS measures:**

| Metric | What it checks | Score range |
|--------|---------------|-------------|
| Faithfulness | Is every claim in the answer supported by the context? | 0–1 |
| Answer Relevancy | Does the answer actually answer the question? | 0–1 |
| Context Precision | Are retrieved chunks actually relevant? | 0–1 |
| Context Recall | Does context cover the ground truth answer? | 0–1 |

**Usage:**
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

result = evaluate(
    Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }),
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)
```

**Synthetic test set generation:**
```python
from ragas.testset.generator import TestsetGenerator
generator = TestsetGenerator.with_openai()
testset = generator.generate_with_langchain_docs(docs, test_size=50)
```

```
requirements:
ragas==0.1.21
datasets==2.20.0
```

---

## 10. Observability

### Structured Logging
```python
import logging
from pythonjsonlogger import jsonlogger

handler = logging.StreamHandler()
handler.setFormatter(jsonlogger.JsonFormatter(
    "%(asctime)s %(name)s %(levelname)s %(message)s"
))
```
Every query logs: query text, latency, token count, cost, cache hit, model used.

### Prometheus Metrics
```python
from prometheus_client import Counter, Histogram, Gauge
query_counter = Counter("rag_queries_total", "Total queries", ["model", "cache_hit"])
query_latency = Histogram("rag_query_latency_seconds", "Query latency")
active_docs = Gauge("rag_documents_total", "Indexed document count")
```
Exposed at `GET /metrics` — scrape-able by Prometheus or Grafana Cloud.

### LangSmith (Optional)
```bash
export LANGSMITH_API_KEY=...
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_PROJECT=docmind
```
Provides full step-by-step trace of every LCEL chain execution with latency breakdown.

### Drift Detection (Evidently AI)
```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Run weekly: compare embedding distribution of last 7 days vs baseline
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=baseline_df, current_data=current_df)
```

```
requirements:
prometheus-client==0.20.0
python-json-logger==2.0.7
evidently==0.4.33
```

---

## 11. Caching

### Development: In-Memory Semantic Cache
```python
from langchain.globals import set_llm_cache
from langchain.cache import InMemorySemanticCache
set_llm_cache(InMemorySemanticCache(embedding=embeddings, score_threshold=0.95))
```
Cached within a single server process lifetime. Resets on restart.

### Production: Redis Semantic Cache
```python
from langchain_community.cache import RedisSemanticCache
set_llm_cache(RedisSemanticCache(
    embedding=embeddings,
    redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
    score_threshold=0.95
))
```
Persists across restarts. Configurable TTL. Cache stats visible in admin panel.

```
requirements:
redis==5.0.7
langchain-redis==0.0.2
```

---

## 12. Configuration Management

### `.env` (copy from `.env.example`)
```dotenv
# LLM
LLM_PROVIDER=openai                          # openai | anthropic | ollama
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OLLAMA_BASE_URL=http://ollama:11434
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=1024

# Embeddings
EMBEDDING_PROVIDER=openai                    # openai | huggingface
EMBEDDING_MODEL=text-embedding-3-small       # or all-MiniLM-L6-v2

# Retrieval
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=20
RERANKER_TOP_K=5
RETRIEVAL_MODE=hybrid                        # similarity | mmr | hybrid
ENABLE_QUERY_EXPANSION=true
ENABLE_HYDE=false
ENABLE_RERANKING=true
ENABLE_CONTEXT_COMPRESSION=false

# Cache
CACHE_BACKEND=memory                         # memory | redis
REDIS_URL=redis://redis:6379
CACHE_TTL_SECONDS=86400

# Observability
LANGSMITH_API_KEY=                           # optional
LANGCHAIN_TRACING_V2=false
LANGCHAIN_PROJECT=docmind
LOG_LEVEL=INFO
ENABLE_QUERY_LOGGING=true

# API
API_KEY_AUTH_ENABLED=false
API_KEY=your-secret-key
CORS_ORIGINS=http://localhost:8501

# Storage
UPLOAD_DIR=data/uploads
FAISS_INDEX_DIR=data/faiss_index
EVAL_RESULTS_DIR=evaluation/results
```

---

## 13. Docker Setup

### `docker-compose.yml`
```yaml
services:
  backend:
    build: .
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    volumes:
      - ./data:/app/data
    env_file: .env
    ports:
      - "8000:8000"
    depends_on:
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    build: .
    command: streamlit run streamlit_app/app.py --server.port 8501
    volumes:
      - ./data:/app/data
    env_file: .env
    ports:
      - "8501:8501"
    depends_on:
      - backend

  redis:
    image: redis:7.2-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  ollama:                          # optional local LLM
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    profiles: ["local-llm"]        # enable with: docker compose --profile local-llm up

volumes:
  redis_data:
  ollama_data:
```

### `Dockerfile`
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps for pytesseract + pdf2image
RUN apt-get update && apt-get install -y \
    tesseract-ocr poppler-utils libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 8501
```

---

## 14. Development Tooling

### `requirements-dev.txt`
```
pytest==8.2.2
pytest-asyncio==0.23.7
pytest-cov==5.0.0
httpx==0.27.0           # async test client for FastAPI
black==24.4.2
isort==5.13.2
ruff==0.4.9
mypy==1.10.1
pre-commit==3.7.1
```

### `.pre-commit-config.yaml`
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.9
    hooks:
      - id: ruff
        args: [--fix]
  - repo: https://github.com/psf/black
    rev: 24.4.2
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.1
    hooks:
      - id: mypy
        additional_dependencies: [pydantic, fastapi]
```

---

## 15. Complete Project Folder Structure

```
docmind/
│
├── .env                          # environment variables (git-ignored)
├── .env.example                  # template with all variables documented
├── .gitignore
├── .pre-commit-config.yaml       # lint/format hooks
├── docker-compose.yml            # full stack: backend + frontend + redis + ollama
├── Dockerfile
├── Makefile                      # shortcuts: make dev, make test, make ingest, make eval
├── README.md                     # setup guide, demo GIF, RAGAS scores table
├── PRD.md                        # this project's PRD (you are reading TECHSTACK.md)
├── TECHSTACK.md                  # this file
├── requirements.txt              # production dependencies (pinned)
├── requirements-dev.txt          # dev/test dependencies
├── pyproject.toml                # black, isort, mypy, ruff config
│
├── app/                          # FastAPI backend
│   ├── __init__.py
│   ├── main.py                   # FastAPI app, lifespan, middleware, router registration
│   ├── config.py                 # Pydantic Settings model — reads from .env
│   ├── dependencies.py           # FastAPI Depends() providers (vectorstore, llm, embeddings)
│   │
│   ├── api/                      # Route handlers
│   │   ├── __init__.py
│   │   ├── documents.py          # POST /documents/upload, GET /documents, DELETE /documents/{id}
│   │   ├── query.py              # POST /query, POST /query/stream (SSE)
│   │   ├── conversations.py      # GET/POST/DELETE /conversations
│   │   ├── evaluate.py           # POST /evaluate, GET /evaluate/results, GET /evaluate/export
│   │   ├── settings.py           # GET/PATCH /settings
│   │   └── health.py             # GET /health, GET /metrics
│   │
│   ├── models/                   # Pydantic v2 schemas
│   │   ├── __init__.py
│   │   ├── document.py           # DocumentRecord, DocumentStatus, ChunkPreview
│   │   ├── query.py              # QueryRequest, QueryResponse, SourceChunk
│   │   ├── conversation.py       # ConversationSession, Message
│   │   └── evaluation.py         # EvaluationRequest, EvaluationResult
│   │
│   └── middleware/
│       ├── __init__.py
│       ├── logging.py            # Request/response logging middleware
│       ├── timing.py             # Latency header injection
│       └── api_key.py            # Optional API key auth middleware
│
├── rag/                          # Core RAG pipeline (framework-agnostic logic)
│   ├── __init__.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py             # File-type router → correct LangChain loader
│   │   ├── splitter.py           # Chunking strategies (recursive, semantic, parent-doc)
│   │   ├── embedder.py           # Embedding model factory (OpenAI / HuggingFace)
│   │   ├── indexer.py            # FAISS + BM25 index build/update/delete
│   │   ├── ocr.py                # pytesseract OCR fallback for scanned PDFs
│   │   └── pipeline.py           # Orchestrates: load → split → embed → index → save
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── faiss_retriever.py    # FAISS dense retriever wrapper
│   │   ├── bm25_retriever.py     # BM25 sparse retriever wrapper
│   │   ├── hybrid.py             # EnsembleRetriever + RRF fusion
│   │   ├── reranker.py           # Cross-encoder reranking (ms-marco-MiniLM)
│   │   ├── query_expansion.py    # MultiQueryRetriever, HyDE
│   │   ├── compressor.py         # LLMChainExtractor context compression
│   │   └── factory.py            # Builds the full retriever from config settings
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── llm_factory.py        # LLM provider factory (OpenAI/Anthropic/Ollama)
│   │   ├── chain.py              # LCEL RAG chain definition
│   │   ├── memory.py             # Conversation history management
│   │   ├── streaming.py          # SSE token streaming helpers
│   │   └── cost_tracker.py       # Token count → cost calculation per provider
│   │
│   ├── cache/
│   │   ├── __init__.py
│   │   └── semantic_cache.py     # InMemory / Redis semantic cache setup
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── ragas_eval.py         # RAGAS metric computation
│       ├── testset_gen.py        # Synthetic Q&A test set generation
│       └── drift.py              # Evidently AI drift detection
│
├── prompts/                      # Versioned prompt templates
│   ├── system_default.txt        # Default system prompt
│   ├── rag_qa.txt                # Main RAG Q&A prompt (context + question → answer)
│   ├── query_rewrite.txt         # Query rewriting prompt
│   ├── hyde.txt                  # HyDE hypothetical document prompt
│   └── compression.txt           # Context compression extraction prompt
│
├── streamlit_app/                # Streamlit multi-page frontend
│   ├── app.py                    # Page 0: Main chat interface
│   ├── utils.py                  # Shared helpers (API client, session state init)
│   ├── components/
│   │   ├── __init__.py
│   │   ├── chat_message.py       # Custom chat message renderer with citations
│   │   ├── source_expander.py    # Source chunk expandable card
│   │   ├── retrieval_sidebar.py  # Sidebar controls (sliders, toggles, selectors)
│   │   └── metrics_card.py       # RAGAS metric card component
│   └── pages/
│       ├── 1_Documents.py        # Document upload + management
│       ├── 2_Evaluation.py       # RAGAS evaluation dashboard
│       ├── 3_Settings.py         # Config management
│       └── 4_About.py            # README / tech stack display
│
├── data/                         # Runtime data (git-ignored)
│   ├── uploads/                  # Uploaded documents (raw files)
│   ├── faiss_index/              # Persisted FAISS + BM25 index
│   │   ├── index.faiss
│   │   ├── index.pkl             # docstore
│   │   └── bm25.pkl              # BM25 index pickle
│   └── metadata.json             # Document registry (id, filename, chunks, status)
│
├── evaluation/
│   ├── testsets/                 # Generated synthetic Q&A datasets
│   │   └── testset_<date>.csv
│   └── results/                  # RAGAS evaluation outputs
│       └── eval_<date>.json
│
├── logs/                         # Runtime logs (git-ignored)
│   └── app.log
│
└── tests/
    ├── __init__.py
    ├── conftest.py               # Fixtures: test docs, mock LLM, test FAISS store
    ├── test_ingestion.py         # Loader, splitter, embedder, indexer
    ├── test_retrieval.py         # FAISS, BM25, hybrid, reranker, query expansion
    ├── test_chain.py             # LCEL chain end-to-end, memory, streaming
    ├── test_cache.py             # Cache hit/miss, TTL
    ├── test_evaluation.py        # RAGAS metric computation on fixtures
    └── test_api/
        ├── test_documents_api.py # Upload, list, delete endpoints
        ├── test_query_api.py     # Query, streaming endpoints
        └── test_evaluate_api.py  # Evaluation endpoints
```

---

## 16. `requirements.txt` (Complete, Pinned)

```
# Core framework
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

# Utilities
python-dotenv==1.0.1
httpx==0.27.0
tenacity==8.4.1            # retry with backoff
numpy==1.26.4
pandas==2.2.2
```

---

## 17. `Makefile`

```makefile
.PHONY: dev test ingest eval lint format docker-up docker-down

dev:
	uvicorn app.main:app --reload --port 8000 &
	streamlit run streamlit_app/app.py --server.port 8501

test:
	pytest tests/ -v --cov=app --cov=rag --cov-report=term-missing

ingest:
	python -c "from rag.ingestion.pipeline import ingest_directory; ingest_directory('data/uploads/')"

eval:
	python -c "from rag.evaluation.ragas_eval import run_evaluation; run_evaluation()"

lint:
	ruff check . && mypy app/ rag/

format:
	black . && isort .

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf data/faiss_index data/uploads logs/
```

---

## 18. Feature–File Mapping (Quick Reference)

| Feature | Files |
|---------|-------|
| Hybrid search (FAISS + BM25 + RRF) | `rag/retrieval/hybrid.py`, `rag/retrieval/bm25_retriever.py` |
| MMR retrieval | `rag/retrieval/faiss_retriever.py` (search_type param) |
| Cross-encoder reranking | `rag/retrieval/reranker.py` |
| Query expansion / multi-query | `rag/retrieval/query_expansion.py` |
| HyDE | `rag/retrieval/query_expansion.py`, `prompts/hyde.txt` |
| Context compression | `rag/retrieval/compressor.py` |
| Parent document retrieval | `rag/retrieval/factory.py` |
| Streaming SSE | `app/api/query.py`, `rag/generation/streaming.py` |
| Conversation memory | `rag/generation/memory.py` |
| Semantic cache (memory) | `rag/cache/semantic_cache.py` |
| Semantic cache (Redis) | `rag/cache/semantic_cache.py` + Redis service |
| RAGAS evaluation | `rag/evaluation/ragas_eval.py`, `app/api/evaluate.py` |
| Synthetic test set | `rag/evaluation/testset_gen.py` |
| Drift detection | `rag/evaluation/drift.py` |
| Cost tracking | `rag/generation/cost_tracker.py` |
| Prometheus metrics | `app/api/health.py` |
| Structured logging | `app/middleware/logging.py` |
| LangSmith tracing | enabled via env vars in `app/config.py` |
| OCR fallback | `rag/ingestion/ocr.py` |
| Multi-provider LLM | `rag/generation/llm_factory.py` |
| API key auth | `app/middleware/api_key.py` |
| OpenAPI docs | FastAPI auto-generation at `/docs` |
| Chat export PDF/MD | `streamlit_app/pages/` + `rag/generation/chain.py` |
| Source citation UI | `streamlit_app/components/source_expander.py` |
| Retrieval control panel | `streamlit_app/components/retrieval_sidebar.py` |
| Chunk preview | `app/api/documents.py` (GET /documents/{id}/chunks) |
| Thumbs up/down feedback | `streamlit_app/components/chat_message.py` |
```
