# Varagity — RAG System Specification

> **Status:** Draft v1 (design)
> **Owner:** @luisegarduno
> **Scope of this document:** the *first* end-to-end version — a terminal-based, Docker-Compose-orchestrated Retrieval-Augmented Generation (RAG) pipeline with **Contextual Retrieval** (contextual embeddings + contextual BM25). It is written to be implemented incrementally and to grow into a much larger system, so **modularity, logging, and tests are first-class requirements, not afterthoughts.**

---

## Table of Contents

1. [Goals & Non-Goals](#1-goals--non-goals)
2. [Background: Vanilla RAG → Contextual Retrieval](#2-background-vanilla-rag--contextual-retrieval)
3. [High-Level Architecture](#3-high-level-architecture)
4. [Technology Stack](#4-technology-stack)
5. [Repository Layout & Modularity Conventions](#5-repository-layout--modularity-conventions)
6. [Configuration (`.env`)](#6-configuration-env)
7. [Infrastructure (`docker-compose`)](#7-infrastructure-docker-compose)
8. [Data Model & Metadata Schema](#8-data-model--metadata-schema)
9. [Ingestion Pipeline](#9-ingestion-pipeline)
10. [Query / Runtime Pipeline](#10-query--runtime-pipeline)
11. [Contextual Retrieval Details](#11-contextual-retrieval-details)
12. [Model Clients](#12-model-clients)
13. [CLI / Terminal UX](#13-cli--terminal-ux)
14. [Logging, Observability & Verbosity](#14-logging-observability--verbosity)
15. [Testing Strategy](#15-testing-strategy)
16. [Evaluation System](#16-evaluation-system)
17. [`golden-docs/` — Architecture Documentation](#17-golden-docs--architecture-documentation)
18. [Coding Conventions & Deviations from the Reference](#18-coding-conventions--deviations-from-the-reference)
19. [Milestones & Definition of Done](#19-milestones--definition-of-done)
20. [Roadmap (Post-v1)](#20-roadmap-post-v1)
21. [Open Questions & Decisions](#21-open-questions--decisions)
22. [References](#22-references)

---

## 1. Goals & Non-Goals

### 1.1 Goals (v1)

- A single `docker compose up` brings up **all** services (LLM, embeddings, vector DB, BM25, orchestration) plus the app.
- The app ingests a corpus from a configurable `docs/` directory (**PDF, `.txt`, `.md`** only in v1) and answers user questions from a terminal prompt.
- **Contextual Retrieval** is implemented end-to-end: each chunk is situated within its parent document by an LLM before being embedded and indexed.
- Every pipeline step (parse → chunk → contextualize → embed → store) is a tracked **Prefect** task/flow.
- **Metadata is captured and persisted** for every chunk (this is a hard requirement — see [§8](#8-data-model--metadata-schema)).
- **Self-hosted models only**: LLM via a local `llama.cpp` server, embeddings via a local **infinity** server. No external API dependency at runtime.
- **Tests for everything** and **rich, structured logging** everywhere.
- Codebase is **modular**: adding a new chunking strategy, parser, or retrieval method = adding one file in a directory, not editing a monolith.

### 1.2 Non-Goals (explicitly deferred to [§20](#20-roadmap-post-v1))

- Re-ranking (cross-encoder / LLM rerank).
- Web/GUI frontend (terminal only for v1).
- Grafana + Prometheus dashboards.
- Non-text modalities (`.mp3`, `.pptx`, images, …) and advanced retrieval (hierarchical index, sentence-window, …).

---

## 2. Background: Vanilla RAG → Contextual Retrieval

**Vanilla RAG**, for reference (this is the baseline we extend):

*Index time*
1. Pre-process documents (extract text from files).
2. Split the corpus into chunks.
3. Embed chunks into vectors.
4. Store vectors in a vector index.
5. Define a **context prompt** template that instructs the LLM to answer using retrieved context.

*Run time*
6. User submits a query.
7. Embed the query with the **same** encoder.
8. Search the query vector against the index.
9. Take top-k.
10. Retrieve the corresponding chunks.
11. Feed the chunks into the context prompt as context, and generate.

**The problem with vanilla RAG:** a chunk embedded in isolation loses the context of its parent document ("the company grew 3%" — *which* company, *which* period?), which causes retrieval misses.

**Contextual Retrieval** (Anthropic) fixes this by prepending, to each chunk, a short LLM-generated blurb that situates the chunk within the whole document *before* embedding **and** before BM25 indexing. Empirically this reduces retrieval failures substantially (≈35% with contextual embeddings alone; ≈49% when combined with contextual BM25; ≈67% when reranking is added — reranking is out of scope for v1). See [§11](#11-contextual-retrieval-details) and [§22](#22-references).

Varagity implements **contextual embeddings + contextual BM25** in v1 (the ≈49% tier), leaving reranking for later.

---

## 3. High-Level Architecture

Microservices, all on one Docker Compose network. The Python **app** is a client of every backing service.

```
                              docker compose network: varagity-net
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │                                                                                │
 │   ┌───────────────────┐   ┌────────────────────┐   ┌───────────────────────┐  │
 │   │  llama.cpp server │   │ infinity-embeddings│   │   PostgreSQL+pgvector  │  │
 │   │  (LLM, GPU)       │   │ (embeddings, GPU)  │   │   (dense vectors)      │  │
 │   │  :8080 /v1        │   │ :8081 /v1          │   │   :5432                │  │
 │   └─────────▲─────────┘   └─────────▲──────────┘   └───────────▲───────────┘  │
 │             │                       │                          │              │
 │   ┌─────────┴───────────────────────┴──────────────────────────┴──────────┐  │
 │   │                          app  (Python / uv)                           │  │
 │   │   ingestion flow  +  query flow  (orchestrated by Prefect)            │  │
 │   └─────────┬───────────────────────┬──────────────────────────┬──────────┘  │
 │             │                       │                          │              │
 │   ┌─────────▼─────────┐   ┌─────────▼──────────┐    (Prefect API :4200)       │
 │   │   Elasticsearch   │   │   Prefect server   │◄─── flow/task run tracking   │
 │   │   (BM25) :9200    │   │   (orchestration)  │                              │
 │   └───────────────────┘   └────────────────────┘                             │
 └──────────────────────────────────────────────────────────────────────────────┘
```

Two logical pipelines run inside **app** (both instrumented with Prefect):

- **Ingestion pipeline** ([§9](#9-ingestion-pipeline)): `discover → parse → chunk → contextualize → embed → store(pgvector + Elasticsearch)`.
- **Query pipeline** ([§10](#10-query--runtime-pipeline)): `query → embed → retrieve(hybrid) → build context prompt → generate → display`.

---

## 4. Technology Stack

| Concern | Choice (v1) | Notes |
|---|---|---|
| Orchestration | **Docker Compose** | One command up. GPU via `nvidia-container-toolkit`. |
| App language | **Python 3.12**, managed by **`uv`** | `uv sync` / `uv run`. |
| Frontend language | TypeScript | **Deferred** — no UI in v1. |
| LLM serving | **`llama.cpp` server** (OpenAI-compatible `/v1`) | Self-hosted, GPU. Model dir bind-mounted (no copy). |
| Embedding serving | **infinity** (`michaelf34/infinity`) | Self-hosted, GPU. Already in repo compose. `multilingual-e5-large-instruct`, **1024-dim**. |
| Dense vector store | **PostgreSQL + `pgvector`** | Replaces the **dropped** Qdrant-GPU plan — see [§21](#21-open-questions--decisions). |
| Sparse / keyword search | **Elasticsearch** (BM25) | Contextual BM25 index. |
| Pipeline orchestration | **Prefect** (v3) | Tracks every step; UI at `:4200`. |
| PDF extraction | **Docling** | Rich structure-aware PDF → text/markdown. |
| Text/Markdown extraction | stdlib read | `.txt` and `.md` share one path. |
| Chunking | `langchain-text-splitters` (`RecursiveCharacterTextSplitter`) | Pluggable — one strategy per file ([§5](#5-repository-layout--modularity-conventions)). |
| LLM/embedding client | `openai` python SDK (points at local servers) | Both servers are OpenAI-compatible. |
| Config | `pydantic-settings` + `.env` | Typed, validated, testable. |
| Console output | **`rich`** | Panels, progress bars, markdown — used for verbosity/debug. |
| Logging | stdlib `logging` + `rich.logging.RichHandler` | Persistent logs; distinct from `verbose` console output ([§14](#14-logging-observability--verbosity)). |
| HTTP / retries | `httpx`, `tenacity` | Retry transient model/DB calls. |
| Testing | `pytest`, `pytest-cov`, `pytest-mock`, `respx`, `testcontainers` | See [§15](#15-testing-strategy). |
| Lint/format/type | `ruff`, `ruff format`, `mypy`, `pre-commit` | |

---

## 5. Repository Layout & Modularity Conventions

The guiding rule: **anything we expect to have "dozens of" is a directory where each file is one implementation, discovered through a registry.** Chunking is the canonical example the user called out; parsers and retrievers follow the same shape.

```
varagity/
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml                 # deps + tool config (ruff/mypy/pytest)
├── .env / .env.example            # single source of config (§6)
├── main.py                        # thin entrypoint -> varagity.cli.app:run
│
├── varagity/                      # the application package
│   ├── __init__.py
│   ├── config.py                  # pydantic-settings Settings (loads .env)
│   ├── logging_setup.py           # configure logging + RichHandler
│   │
│   ├── models/                    # self-hosted model clients (§12)
│   │   ├── registry.py            # getModel(model_type=...) factory
│   │   ├── llm.py                 # llama.cpp chat client
│   │   └── embeddings.py          # infinity embeddings client
│   │
│   ├── ingest/
│   │   ├── discovery.py           # getFilePaths(): scan docs dir, bucket by type
│   │   ├── loader.py              # orchestrates parse→chunk→contextualize
│   │   └── parsers/               # ← one parser per file (modular)
│   │       ├── base.py            # Parser protocol + PARSER_REGISTRY
│   │       ├── text.py            # .txt / .md
│   │       └── pdf.py             # docling
│   │
│   ├── chunking/                  # ← one strategy per file (modular; called out by user)
│   │   ├── base.py                # ChunkingStrategy protocol + CHUNKER_REGISTRY
│   │   └── recursive_character.py # default v1 strategy
│   │
│   ├── context/
│   │   └── contextual.py          # situate_context(): LLM chunk-in-document blurb
│   │
│   ├── stores/
│   │   ├── vector_store.py        # ContextualVectorDB over pgvector
│   │   ├── bm25_store.py          # ElasticsearchBM25
│   │   └── schema.sql             # DDL: extension + tables + indexes
│   │
│   ├── retrieval/                 # ← one method per file (modular)
│   │   ├── base.py                # Retriever protocol + RETRIEVER_REGISTRY
│   │   ├── semantic.py            # dense (pgvector) search
│   │   ├── bm25.py                # sparse (Elasticsearch) search
│   │   └── hybrid.py              # rank fusion of semantic + bm25
│   │
│   ├── generation/
│   │   └── answer.py              # build context prompt + generate answer (§10.2)
│   │
│   ├── pipeline/
│   │   ├── ingest_flow.py         # Prefect @flow: full ingestion
│   │   └── query_flow.py          # Prefect @flow: single query
│   │
│   ├── eval/
│   │   ├── datasets.py            # golden (query -> relevant chunk ids) loaders
│   │   └── evaluate.py            # recall@k / pass@k across retrieval methods
│   │
│   ├── cli/
│   │   └── app.py                 # terminal app: ingest on start, then Q&A loop
│   │
│   └── debug/
│       └── show.py                # v_<name>() rich helpers used by verbose= params
│
├── tests/                         # mirrors varagity/ (§15)
│   ├── unit/
│   ├── integration/
│   └── fixtures/                  # tiny sample .pdf/.txt/.md + golden qa
│
├── data/
│   └── eval/                      # evaluation datasets (golden qa, results)
│
├── docs/                          # INGEST CORPUS (gitignored) — the "docs directory" from .env
└── golden-docs/                   # ARCHITECTURE docs describing the system (§17)
```

> ⚠️ **Naming caution:** the *ingest input* directory is `docs/` (configurable via `DOCS_PATH`, gitignored), while the *architecture documentation* lives in `golden-docs/`. Do not conflate them.

### 5.1 The registry pattern (applies to `parsers/`, `chunking/`, `retrieval/`)

Each pluggable family exposes:

- a `base.py` defining a `Protocol`/ABC (the interface) and a `REGISTRY: dict[str, T]`;
- a `@register("name")` decorator so each implementation self-registers on import;
- a `get(name) -> impl` accessor used by callers, selected by an `.env` value (e.g. `CHUNKING_STRATEGY=recursive_character`, `RETRIEVAL_METHOD=hybrid`).

```python
# chunking/base.py  (sketch)
from typing import Protocol
from langchain_core.documents import Document

class ChunkingStrategy(Protocol):
    def split(self, text: str, *, source_meta: dict, verbose: int = 0) -> list[Document]: ...

CHUNKER_REGISTRY: dict[str, "ChunkingStrategy"] = {}

def register(name: str):
    def deco(cls):
        CHUNKER_REGISTRY[name] = cls()
        return cls
    return deco

def get_chunker(name: str) -> "ChunkingStrategy":
    if name not in CHUNKER_REGISTRY:
        raise KeyError(f"Unknown chunking strategy '{name}'. Available: {list(CHUNKER_REGISTRY)}")
    return CHUNKER_REGISTRY[name]
```

Adding a strategy later (e.g. `semantic.py`, `sentence_window.py`, `markdown_aware.py`) requires **no edits** to callers — only a new file with `@register(...)`.

---

## 6. Configuration (`.env`)

A single `.env` at the repo root is the source of truth. It is consumed twice:
- **Docker Compose** interpolates `${lowercase_vars}` for service definitions.
- **The app** loads the same file via `pydantic-settings` (and the `app` service uses `env_file: .env`).

> **Host vs. container hostnames:** inside the compose network, services address each other by **service name** (`llamacpp`, `infinity-embeddings`, `postgres`, `elasticsearch`, `prefect`). When running the app on the *host* (e.g. during tests), use `localhost`. The example below shows the in-container values; host overrides are commented.

```dotenv
# ─────────────────────────────────────────────────────────────
# Docker Compose interpolation vars (lowercase)
# ─────────────────────────────────────────────────────────────
models_volume="/home/blurry/Desktop/ML/models"                 # bind-mounted into llama.cpp
embeddings_volume="/home/blurry/Desktop/ML/models/embedding"   # bind-mounted into infinity
secret_infinity_key="change-me"

# ─────────────────────────────────────────────────────────────
# Document ingestion
# ─────────────────────────────────────────────────────────────
DOCS_PATH="/app/docs"                    # container path; host: ./docs
ALLOWED_EXTENSIONS=".pdf,.txt,.md"       # v1 whitelist

# ─────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────
CHUNKING_STRATEGY="recursive_character"
CHUNK_SIZE=400                           # NOTE: characters for RecursiveCharacterTextSplitter (see §9.3)
CHUNK_OVERLAP=50

# ─────────────────────────────────────────────────────────────
# Embeddings (infinity)  — 1024-dim for multilingual-e5-large-instruct
# ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL="infloat/multilingual-e5-large-instruct"
EMBEDDING_API_URL="http://infinity-embeddings:8081/v1"   # host: http://localhost:8081/v1
EMBEDDING_API_KEY="${secret_infinity_key}"
EMBEDDING_DIM=1024
EMBEDDING_BATCH_SIZE=32

# ─────────────────────────────────────────────────────────────
# LLM (llama.cpp) — self-hosted
# ─────────────────────────────────────────────────────────────
BASE_MODEL="Qwythos-9B-Claude-Mythos-5-1M-Q8_0.gguf"
BASE_MODEL_API_URL="http://llamacpp:8080/v1"             # host: http://localhost:8080/v1
BASE_MODEL_API_KEY="none"
MAX_TOKENS=8192
LLM_TEMPERATURE=0.6

# ─────────────────────────────────────────────────────────────
# PostgreSQL + pgvector
# ─────────────────────────────────────────────────────────────
POSTGRES_HOST="postgres"                 # host: localhost
POSTGRES_PORT=5432
POSTGRES_DB="varagity"
POSTGRES_USER="varagity"
POSTGRES_PASSWORD="change-me"

# ─────────────────────────────────────────────────────────────
# Elasticsearch (BM25)
# ─────────────────────────────────────────────────────────────
ELASTICSEARCH_URL="http://elasticsearch:9200"            # host: http://localhost:9200
BM25_INDEX_NAME="varagity_contextual_bm25"

# ─────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────
RETRIEVAL_METHOD="hybrid"                # semantic | bm25 | hybrid
TOP_K=10
SEMANTIC_WEIGHT=0.8                      # hybrid rank-fusion weights
BM25_WEIGHT=0.2

# ─────────────────────────────────────────────────────────────
# Prefect + logging
# ─────────────────────────────────────────────────────────────
PREFECT_API_URL="http://prefect:4200/api"                # host: http://localhost:4200/api
LOG_LEVEL="INFO"
DEFAULT_VERBOSE=1                        # 0 off | 1 low | 2 high
```

`varagity/config.py` wraps these in a typed `Settings` object (with validation, e.g. `RETRIEVAL_METHOD ∈ {semantic,bm25,hybrid}`, weights summing to 1.0). Modules read `settings`, never `os.getenv` directly — this makes configuration mockable in tests.

---

## 7. Infrastructure (`docker-compose`)

Six services. GPU services declare the `nvidia` device reservation. Every backing service has a **healthcheck** so `app` (with `depends_on: condition: service_healthy`) starts only once dependencies are ready.

> The existing compose already defines `infinity-embeddings` and `app`. This spec adds `llamacpp`, `postgres`, `elasticsearch`, and `prefect`, plus healthchecks and a shared network. `INFINITY_DEVICE_ID=1` is correct for this host (llama.cpp shares the GPU(s)). The infinity host-port binding (`192.168.86.21:8081:8081`) is intentionally left as-is for now — in-container clients reach it via the `infinity-embeddings` service name regardless — and can be relaxed to `8081:8081` later for portability.

### 7.1 Service summary

| Service | Image | Port(s) | GPU | Volume | Healthcheck |
|---|---|---|---|---|---|
| `llamacpp` | `ghcr.io/ggml-org/llama.cpp:server-cuda` | 8080 | ✔ | `${models_volume}:/models:ro` | `GET /health` |
| `infinity-embeddings` | `michaelf34/infinity:latest-trt-onnx` | 8081 | ✔ | `${embeddings_volume}:/models` | `GET /health` |
| `postgres` | `pgvector/pgvector:pg16` | 5432 | ✗ | `pgdata:/var/lib/postgresql/data` + `schema.sql` init | `pg_isready` |
| `elasticsearch` | `docker.elastic.co/elasticsearch/elasticsearch:9.2.0` | 9200/9300 | ✗ | `esdata:/usr/share/elasticsearch/data` | `GET /_cluster/health` |
| `prefect` | `prefecthq/prefect:3-latest` | 4200 | ✗ | `prefect:/root/.prefect` | `GET /api/health` |
| `app` | local `Dockerfile` (uv) | – | ✗ | `./docs:/app/docs` | – |

### 7.2 `llamacpp` service (the key new piece)

The user runs `llama-server` locally today; the goal is to run it **in a container** with the model directory **bind-mounted** (so the large `.gguf` is never copied into the image). Command mirrors the working local invocation, parameterized by `${BASE_MODEL}`:

```yaml
  llamacpp:
    container_name: llamacpp
    image: ghcr.io/ggml-org/llama.cpp:server-cuda
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ${models_volume}:/models:ro           # no copy — model stays on host disk
    ports:
      - "8080:8080"
    command: >
      -m /models/${BASE_MODEL}
      --host 0.0.0.0 --port 8080
      --jinja -ngl 99 --threads -1 --ctx-size 16384
      --temp 0.6 --min-p 0.0 --top-p 0.95 --top-k 20 --repeat-penalty 1.05
      -ot ".ffn_(up|down)_exps.=CPU"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 15s
      timeout: 5s
      retries: 20
    restart: unless-stopped
```

> Notes: (1) `--mmproj` (multimodal projector) from the local command is **omitted** in v1 since ingestion is text-only; add it back when image modalities land. (2) `--no-webui` is optional. (3) A single llama.cpp server hosts one model at a time in v1; the `.env` `*_MODEL_API_URL` fields all point at `:8080` accordingly.

### 7.3 `postgres` service

`pgvector/pgvector:pg16` ships the `vector` extension. Mount `varagity/stores/schema.sql` into `/docker-entrypoint-initdb.d/` so the extension + tables + HNSW index are created on first boot (see [§8.2](#82-postgresql-schema)).

### 7.4 `elasticsearch` service

Single-node, security disabled (dev only), per the user's target setup:

```yaml
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:9.2.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - ES_JAVA_OPTS=-Xms1g -Xmx1g
    ports: ["9200:9200", "9300:9300"]
    volumes: ["esdata:/usr/share/elasticsearch/data"]
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:9200/_cluster/health || exit 1"]
      interval: 15s
      timeout: 5s
      retries: 20
```

> The `elasticsearch` python client's major version must match the server major version (9.x ↔ 9.x).

---

## 8. Data Model & Metadata Schema

**Storing metadata is a hard requirement.** Every chunk carries a complete, queryable metadata record. Chunks live in **both** stores and must be joinable by a shared identity: `(doc_id, original_index)` — mirroring the Anthropic cookbook's `ContextualVectorDB` / `ElasticsearchBM25` identity keys and enabling hybrid rank fusion ([§11.4](#114-hybrid-search-rank-fusion)).

### 8.1 Chunk metadata (canonical)

| Field | Type | Description |
|---|---|---|
| `doc_id` | str | Stable id per source document = hash(absolute path + content hash). |
| `chunk_id` | str | `f"{doc_id}::{chunk_index}"`. |
| `original_index` | int | Global monotonic chunk index across the corpus (fusion key). |
| `chunk_index` | int | Chunk position within its document. |
| `source` | str | Absolute file path. |
| `file_name` | str | Basename. |
| `file_type` | str | `pdf` / `txt` / `md`. |
| `page` | int? | Page number (PDF; null otherwise). |
| `content` | str | **Original** chunk text. |
| `context` | str | LLM-generated situating blurb ([§11.1](#111-the-contextual-prompt)). |
| `contextualized_content` | str | `context + "\n\n" + content` — the text actually embedded & BM25-indexed. |
| `chunk_size` / `chunk_overlap` | int | Parameters used (provenance). |
| `chunking_strategy` | str | e.g. `recursive_character`. |
| `embedding_model` | str | e.g. `multilingual-e5-large-instruct`. |
| `n_tokens` | int | Token count of `content`. |
| `content_hash` | str | For idempotency / dedup. |
| `created_at` | datetime | Ingestion timestamp. |

This maps directly onto the reference's convention of stuffing derived data into `chunk.metadata` (e.g. `chunk.metadata['context']`, `chunk.metadata['source']`), but promoted to an explicit, validated schema (a `pydantic` `ChunkRecord` model).

### 8.2 PostgreSQL schema

```sql
CREATE EXTENSION IF NOT EXISTS vector;

-- one row per ingested source document (idempotency + provenance)
CREATE TABLE IF NOT EXISTS documents (
    doc_id        TEXT PRIMARY KEY,
    source        TEXT NOT NULL,
    file_type     TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    n_chunks      INT  NOT NULL,
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- one row per chunk
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id                TEXT PRIMARY KEY,
    doc_id                  TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    original_index          INT  NOT NULL,
    chunk_index             INT  NOT NULL,
    content                 TEXT NOT NULL,          -- original
    context                 TEXT,                   -- LLM situating blurb
    contextualized_content  TEXT NOT NULL,          -- embedded/indexed text
    embedding               vector(1024) NOT NULL,  -- EMBEDDING_DIM
    metadata                JSONB NOT NULL,         -- full ChunkRecord (source, page, tokens, …)
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- cosine HNSW index (e5 embeddings are normalized)
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks(doc_id);
```

**Idempotency:** before ingesting a file, compute its `content_hash`; if a `documents` row with the same `doc_id` + `content_hash` exists, skip re-ingestion. This matters because the app re-scans `docs/` on every start ([§9.1](#91-step-1-discovery)).

### 8.3 Elasticsearch index mapping

Mirrors the cookbook's `ElasticsearchBM25`: the analyzed fields are the **contextualized** text and the original content; identity fields are stored but not analyzed.

```json
{
  "settings": {
    "analysis": { "analyzer": { "default": { "type": "english" } } },
    "similarity": { "default": { "type": "BM25" } }
  },
  "mappings": { "properties": {
    "content":                { "type": "text",    "analyzer": "english" },
    "contextualized_content": { "type": "text",    "analyzer": "english" },
    "doc_id":                 { "type": "keyword", "index": false },
    "chunk_id":               { "type": "keyword", "index": false },
    "original_index":         { "type": "integer", "index": false }
  }}
}
```

---

## 9. Ingestion Pipeline

Implemented as a **Prefect flow** (`pipeline/ingest_flow.py`), each stage a `@task` so runs, retries, timings, and logs are visible in the Prefect UI. Mirrors the reference's `main()` → `getFilePaths` → `loadDocuments` → `create_db` shape, generalized and instrumented.

```
ingest_flow(docs_path):
  buckets      = discover_documents(docs_path)          # @task  (§9.1)
  for path in buckets.text_like + buckets.pdf:
      raw      = parse_document(path)                   # @task  (§9.2)
      chunks   = chunk_document(raw)                    # @task  (§9.3)
      chunks   = contextualize_chunks(raw, chunks)      # @task  (§9.4, LLM)
      vectors  = embed_chunks(chunks)                   # @task  (§9.5, infinity)
      store_chunks(chunks, vectors)                     # @task  (§9.6, pgvector + ES)
```

### 9.1 Step 1 — Discovery (`ingest/discovery.py`)

- Read `DOCS_PATH`; if it doesn't exist, fall back to a default and log a warning.
- Recursively glob for allowed extensions; **ignore everything else**.
- **Bucket by extraction path** (this is the user's explicit requirement):
  - `.txt` + `.md` → `text_like` (same extraction code).
  - `.pdf` → `pdf` (needs Docling).

| Extension | Bucket | Parser |
|---|---|---|
| `.txt`, `.md` | `text_like` | `parsers/text.py` |
| `.pdf` | `pdf` | `parsers/pdf.py` (Docling) |
| anything else | — | ignored (logged at DEBUG) |

Signature follows the reference's verbosity convention:
`discover_documents(docs_path: str, verbose: int = 0) -> Buckets`.

### 9.2 Step 2 — Parse (`ingest/parsers/`)

Each parser implements `Parser.extract(path) -> RawDocument(text, source_meta)` and self-registers.

- **`text.py`** — read `.txt`/`.md` as UTF-8; normalize newlines; run `remove_hyphen_space()` (carried over from the reference to fix line-broken words like `frame-\nwork → framework`).
- **`pdf.py`** — use **Docling** (`DocumentConverter().convert(path)`), export to markdown/plain text. Docling preserves structure (headings, tables) far better than `PyPDFLoader`, and its markdown output composes naturally with markdown-aware chunkers later.

### 9.3 Step 3 — Chunk (`chunking/`)

- Default strategy `recursive_character` wraps `RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)`, returning `list[Document]` with metadata seeded (`source`, `file_type`, `chunk_index`, …).
- **Unit note:** `RecursiveCharacterTextSplitter` counts **characters** by default. The reference used `CHUNK_SIZE=300` (chars); this spec defaults to `400/50` (≈12.5% overlap). A future token-based strategy (`length_function=tiktoken`) is a drop-in new file. Document the unit explicitly wherever `CHUNK_SIZE` appears.

### 9.4 Step 4 — Contextualize (`context/contextual.py`)

For each chunk, call the LLM to produce the situating blurb and store it on `chunk.metadata['context']`, exactly as the reference does in `loadDocuments`:

```python
chunk.metadata["context"] = situate_context(document_text, chunk, verbose=v)
chunk.metadata["contextualized_content"] = chunk.metadata["context"] + "\n\n" + chunk.page_content
```

- Uses the **exact Anthropic cookbook prompt** ([§11.1](#111-the-contextual-prompt)).
- LLM responses are passed through `clean_response()` to strip `<think>…</think>` (reasoning models), carried over from the reference.
- **Cost control:** the parent document is identical across all its chunks. Batch a document's chunks together and (where the backend supports it) rely on prompt caching / a single shared document preamble to avoid re-sending the whole doc per chunk. With a local llama.cpp server this is a throughput concern, not a billing one, but the batching structure is the same.

### 9.5 Step 5 — Embed (`models/embeddings.py`)

- Call the **infinity** `/v1/embeddings` endpoint (OpenAI-compatible) with `contextualized_content`, batched (`EMBEDDING_BATCH_SIZE`).
- **e5 prompt formatting matters:** `multilingual-e5-large-instruct` expects passages as raw text and queries wrapped with an instruction, e.g. `f"Instruct: Given a question, retrieve relevant passages.\nQuery: {q}"`. Encapsulate this in the embeddings client so ingestion (passage mode) and query (query mode) format correctly. Getting this wrong silently degrades recall.
- Retry transient failures with `tenacity`.

### 9.6 Step 6 — Store (`stores/`)

- **pgvector:** upsert one `chunks` row per chunk (embedding + full metadata JSONB), and the parent `documents` row.
- **Elasticsearch:** bulk-index the same chunks into the BM25 index ([§8.3](#83-elasticsearch-index-mapping)).
- Both writes are part of the same Prefect task boundary; on partial failure the task fails loudly and is retried.

---

## 10. Query / Runtime Pipeline

`pipeline/query_flow.py` (Prefect flow) + `cli/app.py` (the loop). Mirrors the reference's `workflow()` `state` dict, generalized.

### 10.1 Steps

1. Prompt the user for a query (`rich` prompt).
2. Embed the query with the **same** infinity model, in **query** mode ([§9.5](#95-step-5--embed)).
3. **Retrieve** via the configured `RETRIEVAL_METHOD` ([§11](#11-contextual-retrieval-details)):
   - `semantic` → pgvector cosine top-k;
   - `bm25` → Elasticsearch top-k (matches the user's step-10 "use BM25 for search");
   - `hybrid` → rank fusion of both (**v1 default**).
4. Take **top-k** (default 10) and **display the matches** (source, score, snippet) via `rich`.
5. **Build the context prompt** and **generate** the answer ([§10.2](#102-context-prompt--generation)).
6. Display the answer with cited sources.

State object (carried through the flow, echoing the reference):
`{ "query", "query_vector", "retrieved" (chunks+scores), "formatted_context", "answer" }`.

### 10.2 Context prompt & generation (`generation/answer.py`)

This is the user's requirement #12 — "create a context prompt that tells the LLM to answer the query given the retrieved context." Each retrieved chunk is formatted with its provenance (source + context + content), as the reference's `retrieve_docs` does:

```
[SOURCE]:  {metadata.source}
[CONTEXT]: {metadata.context}
[CONTENT]: {page_content}
```

joined into `formatted_context`, then fed to the LLM:

```text
You are Varagity, a retrieval-augmented assistant.
Answer the user's QUESTION using ONLY the CONTEXT below.
If the answer is not contained in the context, say you don't know — do not fabricate.
Cite the [SOURCE] of any facts you use.

<context>
{formatted_context}
</context>

QUESTION: {query}
ANSWER:
```

Response is post-processed with `clean_response()` and rendered as markdown via `rich`.

---

## 11. Contextual Retrieval Details

This section pins down the Anthropic Contextual Retrieval design as adapted for Varagity. Reference classes: [`ContextualVectorDB`](https://github.com/anthropics/claude-cookbooks/blob/main/capabilities/contextual-embeddings/guide.ipynb) and `ElasticsearchBM25` from the [contextual embeddings guide](https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide).

### 11.1 The contextual prompt

Reproduced verbatim from the cookbook (and already used by the reference's `contextual_embedding`):

```text
<document>
{doc_content}
</document>

Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall
document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
```

The output is stored as `context`, prepended to the chunk to form `contextualized_content`, which is what gets **embedded** and **BM25-indexed**.

### 11.2 `ContextualVectorDB` (adapted to pgvector)

The cookbook's `ContextualVectorDB` is an in-memory store that (a) generates context per chunk, (b) embeds `contextualized_content`, (c) stores vectors + metadata, (d) exposes `search(query, k)` (cosine). **Varagity keeps these responsibilities but backs them with PostgreSQL/pgvector** (`stores/vector_store.py`) so the index is durable, concurrent, and horizontally inspectable via SQL. `search` becomes:

```sql
SELECT chunk_id, doc_id, original_index, content, context, metadata,
       1 - (embedding <=> :qvec) AS score
FROM chunks
ORDER BY embedding <=> :qvec         -- cosine distance
LIMIT :k;
```

### 11.3 `ElasticsearchBM25`

Ported near-verbatim from the cookbook (`stores/bm25_store.py`): `create_index()` ([§8.3](#83-elasticsearch-index-mapping)), `index_documents(docs)` (bulk), and `search(query, k)` (a `multi_match` over `content` + `contextualized_content`, returning `doc_id`, `original_index`, `content`, `contextualized_content`, `score`).

### 11.4 Hybrid search (rank fusion)

`retrieval/hybrid.py` follows the cookbook's fusion: pull the top `k·N` from **each** retriever, combine by weighted rank fusion, dedupe on `(doc_id, original_index)`, return the top-k.

```python
def hybrid_search(query, k, semantic_weight=0.8, bm25_weight=0.2):
    sem = semantic_search(query, k * 10)          # pgvector
    bm  = bm25_search(query, k * 10)              # elasticsearch
    scores = defaultdict(float)
    for rank, r in enumerate(sem): scores[(r.doc_id, r.original_index)] += semantic_weight * 1/(rank+1)
    for rank, r in enumerate(bm):  scores[(r.doc_id, r.original_index)] += bm25_weight     * 1/(rank+1)
    top = sorted(scores, key=scores.get, reverse=True)[:k]
    return hydrate(top)                           # fetch full chunk rows from pgvector
```

Weights come from `.env` (`SEMANTIC_WEIGHT` / `BM25_WEIGHT`, default 0.8 / 0.2). **Reranking is intentionally not applied in v1** (it would slot in after this step).

---

## 12. Model Clients

`models/registry.py` provides the reference's `getModel(model_type=...)` factory, cleaned up. Both the LLM and embedding servers are OpenAI-compatible, so the `openai` SDK is pointed at the local base URLs.

| `model_type` | Backend | Client |
|---|---|---|
| `default` | llama.cpp `:8080/v1` | `openai.OpenAI(base_url=..., api_key="none").chat.completions` |
| `embedding` | infinity `:8081/v1` | `openai.OpenAI(base_url=..., api_key=INFINITY_KEY).embeddings` |
| `reasoning` / `tool` | llama.cpp (future, separate servers/models) | same as `default` |

> **Fixes vs. the reference `util_model.py`:** the reference has a real bug — it branches on the built-in `type` (`if type == "reasoning"`) instead of the `model_type` parameter, so `reasoning`/`tool` never activate. Varagity's registry branches on `model_type` correctly and validates the argument. It also drops `HuggingFaceEmbeddings` (in-process embedding) in favor of the infinity HTTP client, per the self-hosted-services architecture.

---

## 13. CLI / Terminal UX

`cli/app.py` (invoked by root `main.py`). Startup sequence mirrors the user's described flow:

1. Load `settings`; configure logging.
2. Run `ingest_flow(DOCS_PATH)` — with a `rich` progress bar per stage (as the reference does with `rich.progress.Progress`), showing per-file chunk counts and a contextual-embedding sub-progress bar.
3. Enter the Q&A loop: read a query, run `query_flow`, print the **top-10 matches** and the generated answer. `:quit` exits.

Every user-facing step honors a `verbose` level ([§14](#14-logging-observability--verbosity)); at `verbose=2` the retrieved chunks are rendered as `rich` panels with score/source/content/context, exactly like the reference's `v_retrieve_docs`.

---

## 14. Logging, Observability & Verbosity

Three complementary channels — do not confuse them:

1. **`verbose: int` parameter (per function)** — *human-facing console output* for interactive/debug use, rendered with `rich`. Convention carried directly from the reference:
   - `0` = off, `1` = low (names, counts), `2` = high (full metadata, panels).
   - The actual printing lives in `debug/show.py` as `v_<function_name>(...)` helpers (e.g. `v_discover`, `v_retrieve`), keeping business logic clean — exactly the reference's `util_print.py` split.
   - **Every public function takes `verbose: int = DEFAULT_VERBOSE`.** Invalid levels raise `ValueError`.

2. **`logging` (stdlib + `RichHandler`)** — *persistent, leveled logs* (`LOG_LEVEL`), one logger per module (`logging.getLogger(__name__)`). Used for warnings, retries, timings, errors. Configured once in `logging_setup.py`. Third-party noise (e.g. `pypdf`) is quieted, as the reference does.

3. **Prefect run logs** — *pipeline observability*: task state, retries, durations, and inputs/outputs surfaced in the Prefect UI (`:4200`). Use `get_run_logger()` inside tasks so logs attach to the run.

> Grafana + Prometheus (via a Prefect exporter) is **post-v1** ([§20](#20-roadmap-post-v1)); the logging seams above are designed so metrics can be layered on without refactoring.

---

## 15. Testing Strategy

**Tests for everything we implement** is a hard requirement. `tests/` mirrors `varagity/`. Tooling: `pytest`, `pytest-cov`, `pytest-mock`, `respx` (mock the OpenAI/infinity/ES HTTP), `testcontainers` (real Postgres + Elasticsearch for integration), optional `hypothesis`.

### 15.1 Test layers

| Layer | Marker | What | External deps |
|---|---|---|---|
| **Unit** | (default) | Pure logic, fast, isolated. | None (mocked). |
| **Integration** | `@pytest.mark.integration` | Real Postgres/ES via `testcontainers`; real store round-trips. | Docker. |
| **E2E** | `@pytest.mark.e2e` | Full ingest→query over `tests/fixtures/` mini-corpus; LLM/embeddings mocked or a tiny local model. | Docker. |

### 15.2 Representative unit tests (non-exhaustive)

- `discovery`: correct bucketing (`.txt`+`.md`→text_like, `.pdf`→pdf, others ignored); recursive globbing; missing-dir fallback.
- `parsers/text`: UTF-8 read, newline normalization, `remove_hyphen_space` (the reference's regex cases: `frame-\nwork → framework`, `frame- work → framework`).
- `parsers/pdf`: Docling extraction against a tiny committed fixture PDF.
- `chunking/recursive_character`: chunk count, overlap correctness, metadata seeding, determinism.
- `context/contextual`: prompt is formatted exactly; `clean_response` strips `<think>…</think>`; blurb lands in `metadata['context']`.
- `models/embeddings`: e5 query-vs-passage formatting; batching; `respx`-mocked infinity responses; retry on 5xx.
- `models/registry`: `model_type` dispatch (**including a regression test for the reference's `type` bug**); invalid type raises.
- `stores/vector_store` (integration): upsert + cosine search returns expected order; idempotent re-ingest via `content_hash`.
- `stores/bm25_store` (integration): index mapping created; `multi_match` search ranks the planted document first.
- `retrieval/hybrid`: fusion math + dedupe on `(doc_id, original_index)` with synthetic ranked lists (no services).
- `generation/answer`: context formatting; "not in context → I don't know" behavior with a stubbed LLM.
- `config`: validation (bad `RETRIEVAL_METHOD`, weights not summing to 1) fails fast.

### 15.3 Conventions & CI

- `pytest.ini`/`pyproject`: register markers; `--cov=varagity`; a coverage floor (e.g. ≥80%) that ratchets up.
- Fixtures: a `settings` fixture with test config; `testcontainers` session-scoped Postgres/ES fixtures; a fake-LLM/fake-embeddings fixture.
- **`pre-commit`**: `ruff`, `ruff format`, `mypy`, and fast unit tests. (Fulfills the README "Create pre-commit rules" TODO.)
- CI runs unit tests on every push; integration/e2e on demand or nightly (they need Docker).

---

## 16. Evaluation System

`eval/` provides an offline harness to measure retrieval quality and compare methods — modeled on the cookbook's evaluation (recall@k over a golden set of `(query → relevant chunk ids)`).

- **Dataset** (`data/eval/`): JSONL of `{query, relevant_doc_ids/chunk_ids}`. Seed from `tests/fixtures/` and grow over time.
- **Metrics**: `recall@k` (primary; fraction of relevant chunks retrieved in top-k) and `pass@k`; report per-method.
- **Comparison matrix** — run the *same* queries through each configuration to quantify the Contextual Retrieval uplift:
  1. semantic, **non**-contextual embeddings (baseline);
  2. semantic, contextual embeddings;
  3. contextual BM25 only;
  4. **hybrid** contextual (embeddings + BM25) — the v1 target.
- **Output**: a `rich` table + a persisted results file under `data/eval/`, so regressions are visible when chunking/models change.
- Runs are themselves a Prefect flow (tracked, repeatable). This directly supports future decisions ("did switching chunkers help?").

---

## 17. `golden-docs/` — Architecture Documentation

A living set of docs describing the system as-built (distinct from this forward-looking spec):

- `golden-docs/architecture.md` — the service diagram, data flow, identity keys, and why (Contextual Retrieval rationale).
- `golden-docs/data-model.md` — the metadata schema, pg tables, ES mapping ([§8](#8-data-model--metadata-schema)).
- `golden-docs/pipelines.md` — ingestion & query flows with the Prefect task graph.
- `golden-docs/adr/` — Architecture Decision Records (e.g. *ADR-001: pgvector over Qdrant*, *ADR-002: infinity over FastEmbed*), capturing the decisions in [§21](#21-open-questions--decisions).
- `golden-docs/runbook.md` — operating the stack (bring-up order, healthchecks, resetting volumes, GPU/device-id gotchas).

---

## 18. Coding Conventions & Deviations from the Reference

The reference (`/home/blurry/Desktop/ML/RAG-Research/Demos/Demo-ContextualRetrieval`) establishes the **patterns we keep**:

- Thin `main.py` → discover → load/chunk → build store → workflow.
- `verbose: int` on every function; debug rendering isolated in `util_print.py`-style helpers.
- `rich` for all human-facing output (panels, progress, markdown).
- `getModel(model_type=...)` factory; `state` dict threaded through the query workflow.
- The exact contextual prompt and `clean_response` `<think>`-stripping.
- Storing derived data in chunk metadata (`context`, `source`, …).

**Deliberate changes** (and *why*):

| Reference | Varagity v1 | Why |
|---|---|---|
| FAISS (in-process) | PostgreSQL + pgvector | Durable, concurrent, SQL-inspectable; user requirement. |
| `HuggingFaceEmbeddings` in-process | infinity HTTP client | Self-hosted **service**; GPU isolation; already in compose. |
| `PyPDFLoader` / `UnstructuredImageLoader` | **Docling** for PDF | Structure-aware extraction; markdown output. |
| PDF + **JPG** | PDF + **txt + md** | Matches v1 corpus scope (images deferred). |
| No BM25 | **Contextual BM25** (Elasticsearch) | Completes Contextual Retrieval (≈49% tier). |
| Ad-hoc `os.getenv` | `pydantic-settings` `Settings` | Typed, validated, testable. |
| Imperative `main()` | **Prefect** flows/tasks | Observability + retries per user requirement. |

**Bugs in the reference to *not* copy (add regression tests):**
- `util_model.py` branches on the built-in `type` instead of `model_type` → `reasoning`/`tool` are dead code.
- `main.py` `user_docs_path -= '/'` on a `str` raises `TypeError`; use `rstrip('/')`.

---

## 19. Milestones & Definition of Done

### 19.1 v1 Definition of Done

- [ ] `docker compose up` starts all six services; healthchecks pass; `app` waits for dependencies.
- [ ] llama.cpp runs in-container with the model **bind-mounted** (not copied).
- [ ] Ingesting `docs/` (PDF + txt + md) populates **both** pgvector and Elasticsearch with **full metadata**.
- [ ] Each ingestion stage is a tracked Prefect task; runs visible at `:4200`.
- [ ] Contextual blurb generated per chunk and used for **both** embedding and BM25 indexing.
- [ ] Terminal Q&A: query → hybrid retrieve → top-10 shown → grounded answer with sources.
- [ ] `verbose` levels + `logging` + Prefect logs all working.
- [ ] Unit tests green with coverage ≥ target; integration tests pass against `testcontainers`.
- [ ] Evaluation harness runs and reports recall@k for the four configurations.
- [ ] `golden-docs/` describes the system; `pre-commit` configured.

### 19.2 Suggested build order (each step shippable + tested)

1. **Scaffold**: package layout, `config.py`, `logging_setup.py`, `pyproject` deps, `pre-commit`. *(tests: config)*
2. **Compose**: add `llamacpp`, `postgres`, `elasticsearch`, `prefect`; healthchecks; `schema.sql`. *(smoke: bring-up)*
3. **Model clients**: `getModel` LLM + embeddings (e5 formatting). *(tests: registry, embeddings mocked)*
4. **Discovery + parsers**: bucketing, text, Docling PDF. *(tests: discovery, parsers)*
5. **Chunking**: registry + `recursive_character`. *(tests: chunking)*
6. **Contextualize**: prompt + `clean_response`. *(tests: contextual)*
7. **Stores**: pgvector + ElasticsearchBM25 + metadata. *(integration tests)*
8. **Ingestion flow**: wire 4–7 into a Prefect flow; idempotency. *(e2e on fixtures)*
9. **Retrieval**: semantic, bm25, hybrid fusion. *(tests: hybrid math; integration)*
10. **Generation + CLI**: context prompt, answer, Q&A loop. *(e2e)*
11. **Evaluation**: golden set + recall@k matrix. *(tests: metrics)*
12. **golden-docs/** + ADRs.

---

## 20. Roadmap (Post-v1)

- **Re-ranker** (cross-encoder or LLM) inserted after hybrid fusion.
- **Grafana + Prometheus** via a Prefect-Prometheus exporter; dashboards for ingestion/query latency, recall.
- **Frontend UI** (TypeScript) over a thin API layer.
- **More retrieval methods** (each a file in `retrieval/`): Hierarchical Index Retrieval, Sentence-Window Retrieval, parent-document, HyDE, multi-query.
- **More chunking strategies** (each a file in `chunking/`): semantic, markdown-aware, token-based.
- **More parsers** (each a file in `parsers/`): `.pptx`, `.docx`, `.html`, audio (`.mp3` via ASR), images (re-add llama.cpp `--mmproj`).
- Prompt-cache-aware contextualization batching; embedding-cache; incremental/streaming ingestion.
- Alternative vector-store / embedding backends behind the store & embedding interfaces, **only if a concrete need arises** (the boundaries exist; Qdrant-GPU / FastEmbed are *not* currently planned — see [§21](#21-open-questions--decisions)).

Recommended reading before extending: the **jxnl RAG series** — https://jxnl.co/writing/2025/09/11/rag-series-index/ .

---

## 21. Open Questions & Decisions

| # | Topic | Decision (v1) | Status / note |
|---|---|---|---|
| 1 | **Vector store** | pgvector ✅ | **Confirmed** — Qdrant-GPU is **dropped** (no longer the plan). pgvector is the vector store. Recorded in ADR-001. `README.md` / `CLAUDE.md` still list Qdrant-GPU and should be updated. |
| 2 | **Embedding backend** | infinity ✅ | **Confirmed** — FastEmbed is **dropped** (no longer the plan). infinity is the embedding backend (already in compose). ADR-002. `README.md` / `CLAUDE.md` still list FastEmbed and should be updated. |
| 3 | **Chunk size** | `CHUNK_SIZE=400` chars, overlap `50` (`RecursiveCharacterTextSplitter`) | **Confirmed** — default **400** chars (≈12.5% overlap). Counts characters, not tokens; a token-based strategy stays a future drop-in file. |
| 4 | **Default retrieval in v1** | `hybrid` ✅ | **Confirmed.** Note: the original step-10 narrative said "use BM25"; hybrid is the agreed default, with pure `bm25` / `semantic` still selectable via `RETRIEVAL_METHOD`. |
| 5 | **Multimodal now?** | No — `--mmproj` **omitted** ✅ | **Confirmed** — v1 is text-only. Re-add the projector when image ingestion lands. |
| 6 | **GPU device ids** | infinity on device `1` ✅ | **Confirmed correct for this host** (`INFINITY_DEVICE_ID=1`); llama.cpp shares the GPU(s). Revisit only if deployed on a different host. |
| 7 | **infinity port binding** | keep `192.168.86.21:8081:8081` for now | **Deferred** — fine as-is for now; in-container clients reach infinity via the `infinity-embeddings` service name regardless of the host binding. Relax to `8081:8081` later for portability. |
| 8 | **Prefect deployment model** | server + in-app flows ✅ | **Confirmed** — v1 runs flows from the app against the Prefect API; **no separate worker/agent needed yet.** |
| 9 | **Reasoning/tool models** | single llama.cpp server, one model ✅ | **Confirmed** — multi-model serving deferred (expected to be straightforward to add later). |

---

## 22. References

- Anthropic — **Introducing Contextual Retrieval**: https://www.anthropic.com/news/contextual-retrieval
- Anthropic Cookbook — **Contextual Embeddings guide** (`ContextualVectorDB`, `ElasticsearchBM25`, evaluation): https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide · https://github.com/anthropics/claude-cookbooks/blob/main/capabilities/contextual-embeddings/guide.ipynb
- **jxnl RAG series** (recommended pre-reading): https://jxnl.co/writing/2025/09/11/rag-series-index/
- **Docling** (PDF extraction): https://github.com/docling-project/docling
- **infinity** (embedding server): https://github.com/michaelfeil/infinity
- **pgvector**: https://github.com/pgvector/pgvector
- **Prefect** (v3): https://docs.prefect.io
- **llama.cpp server**: https://github.com/ggml-org/llama.cpp
- **Elasticsearch**: https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html
- **uv**: https://github.com/astral-sh/uv
- Reference implementation (patterns): `/home/blurry/Desktop/ML/RAG-Research/Demos/Demo-ContextualRetrieval`
```
