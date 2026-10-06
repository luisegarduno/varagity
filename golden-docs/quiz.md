---
search:
  exclude: true
hide:
  - navigation
---

# Quiz

You found it. Thirty system-design questions about Varagity's **primary RAG
pipeline** — the Contextual Retrieval path from a file in `docs/` to a cited
answer — asked the way an interviewer would ask them. Answer out loud first,
then reveal. Every solution is **as-built** and names the files and ADRs behind
it, so a mismatch is a pointer to go read the source, not a trick.

The message graph ([ADR-017](adr/ADR-017-graphrag-engine.md)) is deliberately
out of scope for now; it gets its own set later.

<div class="quiz" markdown>

## Services & topology

### 1. What tools make up the observability stack, and how do metrics flow between them?

??? question "Reveal solution"

    Three pieces, all provisioned by compose ([ADR-007](adr/ADR-007-observability-stack.md)):

    - **In-app `prometheus_client` collectors** inside the API process
      (`varagity/observability/metrics.py`): per-stage query latency, retrieval
      score by rank, rerank delta, query/ingest counters, contextualize latency,
      LLM tokens, `dependency_up`, plus the store-derived corpus gauges
      ([ADR-013](adr/ADR-013-corpus-gauges-vs-counters.md)). Exposed by a plain
      `GET /metrics` route on the API — `METRICS_ENABLED` gates only the route;
      recording is unconditional.
    - **Prometheus** (`:9090`) scrapes `api:8000/metrics` every 15 s.
    - **Grafana** (`:3001`) is provisioned from files: one datasource with the
      load-bearing uid `prometheus`, three read-only dashboards (**Query**,
      **Ingestion**, **Infra**), anonymous Viewer access — zero click-ops,
      dashboards as code.

    Two optional exporters ride compose **profiles**, off by default:
    `prefect-exporter` (flow/task-run states polled from the Prefect API, with
    `OFFSET_MINUTES=1440`) and `dcgm-exporter` (GPU VRAM/utilization). Their
    scrape targets are configured permanently, so a disabled profile simply reads
    `up == 0`.

    Prefect's UI (`:4200`) is the *run-tracking* surface — flow/task states,
    durations, logs — not a metrics source: flows run in-process, so the Prefect
    server never sees stage latencies or scores, which is why in-app collectors
    are primary and the exporter is optional.

    Two cross-cutting facts: metrics are **per-process** (only flows run inside
    the API process reach the scrape — a CLI ingest never does), and
    `varagity_dependency_up` is refreshed by the api container's own
    `/api/health` healthcheck every 15 s, so there is no poller.

### 2. How are embeddings produced and served?

??? question "Reveal solution"

    - By **infinity** (`michaelf34/infinity:0.0.76-trt-onnx`, tag pinned because
      upstream does not CI-build `latest-trt-onnx`) in the `infinity-embeddings`
      container on **GPU 1**, serving `multilingual-e5-large-instruct`
      (**1024-dim**) behind an OpenAI-compatible `/v1/embeddings`. The same
      container serves `bge-reranker-v2-m3` at `/v1/rerank` (semicolon
      multi-model syntax) — one GPU service, two models.
    - The client is `EmbeddingsClient` (`varagity/models/embeddings.py`): the
      `openai` SDK pointed at `EMBEDDING_API_URL` with the SDK's own retries
      **disabled** so `tenacity` is the single retry layer (4 attempts,
      exponential backoff on connection errors, 5xx, and 429); batches by
      `EMBEDDING_BATCH_SIZE` (32); warns at ≥480 tokens because e5 truncates at
      512.
    - It owns the **asymmetric e5 formatting** (question 17): passages raw,
      queries `Instruct: …\nQuery: …` — both modes in one client so no caller
      can format inconsistently.
    - Why a service rather than in-process FastEmbed
      ([ADR-002](adr/ADR-002-infinity-over-fastembed.md)): the app container
      stays CPU-only and thin, GPU placement becomes a compose concern, and it
      is the same SDK client pattern as llama.cpp — one library, two base URLs.
    - Operational constraints: the engine must stay `optimum` (torch has no
      `sm_120` kernels for the RTX 5060), the reranker's ONNX is pre-exported,
      `INFINITY_BATCH_SIZE='32;4'` caps the reranker's batch on the 8 GB card,
      GPU pinning is `device_ids: ["1"]` at the Docker layer (the optimum engine
      ignores `INFINITY_DEVICE_ID`), and the host port binds one specific
      interface, so `localhost:8081` does not reach it. The served model name is
      `infloat/multilingual-e5-large-instruct` — typo and all, verbatim.

### 3. Name the ten default compose services. Which one is the only thing a browser talks to, and what gates bring-up?

??? question "Reveal solution"

    | Service | Port | Role |
    |---|---|---|
    | `llamacpp` | 8080 | Chat LLM for answers and situating blurbs (GPU 0) |
    | `infinity-embeddings` | 8081 | e5 embeddings + bge reranker (GPU 1) |
    | `postgres` | 5432 | pgvector: vectors + chunk metadata, conversations, settings |
    | `elasticsearch` | 9200 | Contextual BM25 index |
    | `prefect` | 4200 | Flow/task-run tracking UI (SQLite) |
    | `api` | 8000 | FastAPI: SSE chat, CRUD, uploads/ingest, `/metrics` |
    | `web` | 3000 | The Next.js GUI |
    | `prometheus` | 9090 | Scrapes the API |
    | `grafana` | 3001 | Provisioned dashboards |
    | `app` | — | The CLI (`ingest` / `chat` / `eval`) |

    Plus the two profile-gated exporters, `prefect-exporter` and `dcgm-exporter`.

    - Edges: the browser talks to **`web`** only (and, optionally, Grafana);
      `web` talks only to `api`; `api` and `app` reach the five backing
      services as peer front-ends; Prometheus scrapes `api`. The web app holds
      no pipeline logic — it is a pure JSON + SSE client, which is what keeps
      the frontend swappable.
    - Bring-up is `docker compose up -d --wait`: `app` and `api` declare
      `depends_on: condition: service_healthy` on all five backing services,
      `web` waits on `api`, `grafana` on `prometheus`. Healthcheck quirks worth
      knowing: llama.cpp's `/health` returns 503 while the model loads (~30 s;
      generous retries cover it), infinity's `/health` is outside the `/v1`
      prefix, a single-node Elasticsearch is `yellow` by design (never gate on
      green), and prefect/api probe with python's urllib because their images
      ship no curl. `scripts/smoke.sh` then checks substance — schema tables,
      a 1024-dim embedding, the metric families, the three dashboards.

### 4. How are the two GPUs allocated, and what constraints follow from that?

??? question "Reveal solution"

    - **GPU 0** (RTX 2080 Ti) → `llamacpp` via `count: 1`, which grabs the first
      device. Roughly 9 GB steady state because the configured model is a MoE
      and `-ot ".ffn_(up|down)_exps.=CPU"` offloads the expert FFN weights to
      CPU — which is also why prompt evaluation is noticeably slower than decode
      (~54 tok/s): the CPU-offload signature, not a bug.
    - **GPU 1** (RTX 5060, 8 GB) → `infinity-embeddings` via `device_ids: ["1"]`;
      e5 and the reranker share it, batch-capped (`'32;4'`) to fit.
    - There is **no VRAM isolation between containers** — both see real device
      memory, so the total must fit.
    - llama.cpp runs `--parallel 1` deliberately: newer server builds default to
      four slots sharing one `--ctx-size` KV pool, and two ~8k contextualization
      prompts on different slots exhausted it mid-decode. One slot means the
      full window per request plus stable prompt-cache reuse across a
      document's chunks.
    - `LLM_CONTEXT_TOKENS` (16384) is compose-interpolated into `--ctx-size`
      *and* read by the app, so the two cannot drift; `LLMClient` clamps every
      generation cap against it because llama.cpp hard-500s when prompt plus
      generation reach the window instead of stopping gracefully.

### 5. What is the registry pattern, and where is it used?

??? question "Reveal solution"

    - Any family expected to grow "dozens of" implementations is a directory
      where each file is one implementation, self-registered with
      `@register("name")` on package import and selected by an `.env` value.
      Adding one means one new file plus its import line in the package
      `__init__` — zero caller edits.
    - Families: **parsers** (`text`, `pdf`, `office`, `web`, `image` — selected
      by discovery bucket), **chunking strategies** (`CHUNKING_STRATEGY`),
      **retrievers** (`RETRIEVAL_METHOD`: `semantic`, `bm25`, `hybrid`,
      `reranked`, `hyde`), and **chat engines** (`CHAT_ENGINE`: `simple`,
      `condense_context`). The graph corpus adds two more, out of scope here.
      OCR engines are a *factory* in `parsers/pdf.py` (deliberately not a
      registry), and model clients resolve by `model_type` (`embedding`,
      `rerank`, `default` plus the `reasoning`/`tool` aliases).
    - Shared machinery stays a plain module: the Docling core
      (`parsers/docling_base.py`) is unregistered; only *selectable*
      implementations register.
    - The vocabulary is **hard-coded again** in the `config.py` validators
      (importing the registry there would be circular), each with a
      tuple↔registry regression test so the two grow in lockstep.
    - Composition rides the same seam: `reranked` and `hyde` resolve their base
      retriever through `get_retriever()`, so fusion logic exists exactly once.

## Ingestion

### 6. Walk through what happens when a document is uploaded from the GUI until it is answerable.

??? question "Reveal solution"

    1. **Upload.** The composer 📎 (files *and* folders) or the corpus page sends
       a multipart `POST /api/documents`. Batch caps are checked before any byte
       is written (`UPLOAD_MAX_FILES` 500, `UPLOAD_MAX_TOTAL_MB` 2048 → 422).
       Per file: the client-declared relative path passes the layered rules (no
       absolute paths, `..`, or dotfile segments; reject-don't-transform per
       segment; depth ≤ `UPLOAD_MAX_PATH_DEPTH`; a `resolve()`-containment
       backstop at the write site), the extension is whitelisted
       (`ALLOWED_EXTENSIONS`), bytes stream in 1 MiB slices to a
       `.upload-partial` file (aborting at `UPLOAD_MAX_MB` 50) and move into
       `DOCS_PATH` — folder structure preserved, because the relative path is
       part of `doc_id`. The browser's `File.lastModified` restamps the file's
       mtime so `file_modified_at` carries the document's clock. Per-file
       outcomes ride a 201; the route itself never ingests
       ([ADR-012](adr/ADR-012-relative-path-uploads.md)).
    2. **Trigger.** The composer fires `POST /api/ingest {reingest:false}`; the
       API answers 202 with a run handle — or `409 ingest_already_running`,
       which the composer queues client-side and re-issues when the in-flight
       run's terminal frame arrives. `IngestRunner` starts the *same*
       `ingest_flow` the CLI runs on a daemon thread inside the API process,
       wrapping the flow's `TASK_STAGES` with SSE emitters;
       `GET /api/ingest/status` replays the feed from frame one.
    3. **The flow, per file.** `discover` buckets files by parser family → the
       idempotency check skips any file whose `(doc_id, content_hash)` already
       exists in `documents` → `parse` (Docling; PDFs take the two-pass OCR
       path) → the empty-extraction guard (fewer than 50 non-whitespace
       characters → a 0-chunk `documents` row, never a silent drop) → `chunk`
       (`CHUNKING_STRATEGY`) → `contextualize` (one LLM blurb per chunk,
       sequential so llama.cpp reuses its prompt cache — the ingest's long pole,
       ticked per chunk on the progress bar) → `embed` (`contextualized_content`
       in e5 passage mode, batched) → `store` (Elasticsearch bulk first, then
       one pgvector transaction writing the `documents` row and the chunks).
    4. **Answerable.** The chunk now lives in both stores joined by
       `(doc_id, original_index)`, and the next question fuses it. The corpus
       gauges show it on the next scrape; the ingest counters only because the
       run happened in the API process; and because the run was
       `reingest=false`, the stale-corpus flag is untouched.

### 7. How is a document's identity derived, and what is the cross-store join key?

??? question "Reveal solution"

    ```text
    content_hash   = sha256(file_bytes)                                 # bytes, not parsed text
    doc_id         = sha256(f"{relative_path}:{content_hash}")[:16]     # path relative to DOCS_PATH
    chunk_id       = f"{doc_id}::{chunk_index}"                         # pg primary key, ES _id
    original_index = global monotonic chunk counter, allocated per ingest run
    ```

    - `content_hash` hashes raw **bytes**: unchanged files are skipped *before*
      paying the parse cost, and `doc_id` stays stable across OCR engines.
    - `doc_id` hashes the path **relative to `DOCS_PATH`**
      ([ADR-003 §4](adr/ADR-003-vertical-build-and-ops-choices.md); the spec
      said absolute). Absolute paths differ between host (`/home/…/docs/a.md`)
      and container (`/app/docs/a.md`) and across machines, which would break
      idempotency and make golden eval sets non-portable. The absolute path
      survives only as `source` provenance. Corollary: moving a file within the
      corpus makes it a new document.
    - `(doc_id, original_index)` is the fusion/join identity across pgvector
      and Elasticsearch. `original_index` starts at
      `SELECT COALESCE(MAX(original_index), -1) + 1` and increments in-process.
      A **unique index** enforces the pair in Postgres, so an ingest bug fails
      loudly at write time instead of silently corrupting hybrid fusion.
    - Because the derivation is pure, the golden eval set resolves to
      `chunk_id`s from corpus files alone — no store round-trip.

### 8. What makes ingestion idempotent, and when do you have to reingest?

??? question "Reveal solution"

    - A file whose `(doc_id, content_hash)` already exists in `documents` is
      skipped before parsing; a known 0-chunk document is re-warned, not
      re-parsed.
    - Writes are upserts — `ON CONFLICT (chunk_id) DO UPDATE` in pg, documents
      addressed by `chunk_id` in Elasticsearch — and each document's pg write is
      one transaction, so a partial failure leaves no idempotency marker and the
      next run re-attempts the file.
    - Pipeline-setting changes (`CONTEXTUALIZE`, `CHUNKING_STRATEGY`,
      `CHUNK_SIZE`/`CHUNK_OVERLAP`, `OCR_ENGINE`) do **not** change content
      hashes, so unchanged files stay skipped until `main.py ingest --reingest`
      or `POST /api/ingest` with `reingest=true`. Both run the same flow, which
      deletes each discovered document from **both** stores (ES
      `delete_by_query` first, then the pg cascade) before ingesting it fresh.
    - The GUI surfaces the footgun as a persisted flag: changing a
      reingest-affecting setting on a non-empty corpus sets `_corpus_stale` in
      `app_settings` ("Re-ingest to apply"). **Only a completed API-driven
      `reingest=true` run clears it** — a CLI reingest, patching the setting
      back, or a composer upload (`reingest=false`) never do.
    - Removing a file from `docs/` does not remove its chunks;
      `DELETE /api/documents/{doc_id}` (and the bulk `POST …/delete`) is the
      GUI-driven GC, optionally unlinking the source file when it lives inside
      `DOCS_PATH`.

### 9. Why does `store_chunks` write Elasticsearch first and pgvector last?

??? question "Reveal solution"

    - The pgvector `documents` row is the **idempotency marker**, so it must
      commit last. A crash between the two writes leaves no marker, the file is
      re-attempted on the next run, and the ES bulk — addressed by the
      deterministic `chunk_id` — overwrites rather than duplicates.
    - The pg half (`documents` row plus all chunks) is one transaction, so there
      is never a half-written document behind a marker.
    - The same ordering holds everywhere the stores change together: reingest
      deletes BM25 first (a failed ES delete leaves the marker intact and the
      next run retries both), and `DELETE /api/documents` deletes ES first and
      pg last — a marker deleted first would strand invisible ES chunks, the
      exact v1 gap the route exists to close. The bulk delete applies the same
      order set-wise, one round trip per store.
    - Both writes share one stage boundary, and the `store` task carries
      `retries=2`, which is safe precisely because both writes are idempotent.

### 10. Which parser families exist, and how does PDF extraction decide whether to OCR?

??? question "Reveal solution"

    - Discovery buckets files by extension (under `ALLOWED_EXTENSIONS`) into
      five registered parsers: `text` (`.txt`/`.md`/`.rst`), `pdf`, `office`
      (the OOXML families including macro/template variants, `.csv`,
      OpenDocument), `web` (`.html`/`.htm`/`.xhtml`), and `image` (bitmaps).
      `office`, `web`, and `image` share the *unregistered* Docling core
      (`parsers/docling_base.py`: convert → structure-aware markdown → hyphen
      repair → per-page character counts → `RawDocument`).
    - PDF is **two-pass, inside Docling**
      ([ADR-003 §5](adr/ADR-003-vertical-build-and-ops-choices.md)): pass 1
      converts with `do_ocr=False`; pass 2 re-converts with `do_ocr=True` when
      the result has fewer than `PDF_OCR_MIN_CHARS` (50) non-whitespace
      characters, at least `PDF_OCR_TEXTLESS_PAGE_RATIO` (0.2) textless pages,
      or pass 1 raised. `PDF_OCR_FORCE_FULL_PAGE` skips pass 1 for corrupt text
      layers (garbage text passes the content triggers by definition). Both
      passes share one pipeline, so the fallback changes *how* text is
      recovered, never its downstream shape.
    - The OCR engine is a factory keyed by `OCR_ENGINE`: `easyocr` (the
      benchmark-decided default — [ADR-004](adr/ADR-004-ocr-engine-choice.md):
      error-free where Tesseract drops words) or `tesseract` (~5× faster).
      CPU-only by design; both installed in both images.
    - Provenance rides `extraction` on every chunk: `text`, `ocr_fallback` (PDF
      pass 2), or `ocr` (the image parser, where OCR is the only path). Office
      and web never OCR — their text is digital by construction. A document
      where even OCR finds nothing ends in the empty-extraction guard: a
      0-chunk row and a warning, never a silent drop.
    - `page` is document-level (the first page that contributed text); `.pptx`
      slides and `.xlsx`/`.ods` sheets are Docling pages, so the number rides
      `page`; `.docx` and `.html` expose no pagination.

### 11. What is Contextual Retrieval here, concretely? Which text gets embedded and indexed?

??? question "Reveal solution"

    - Per chunk at ingest, `situate_context()` (`varagity/context/contextual.py`)
      shows the LLM the *whole document* plus the chunk under the verbatim
      Anthropic cookbook prompt and receives a short situating blurb
      (`context`), post-processed with `clean_response()` to strip a reasoning
      model's `<think>` block.
    - `contextualized_content = context + "\n\n" + content` is what gets
      **embedded** (pgvector) **and BM25-indexed** (Elasticsearch). The original
      `content` is preserved alongside and is what the reranker and the answer
      prompt see. So the BM25 index is contextual from its first document.
    - A document's chunks are contextualized sequentially so every call shares
      an identical prompt prefix and llama.cpp reuses its KV/prompt cache — a
      throughput concern with a local server, not a billing one. A fixed chunk
      allowance keeps the truncated document preamble byte-identical across a
      document's chunks.
    - The preamble is budgeted at `LLM_CONTEXT_TOKENS − CONTEXTUALIZE_MAX_TOKENS
      − 2048 − 640` tokens (≈11.6k on the shipped config); longer documents are
      truncated with a warning. Generation runs under `CONTEXTUALIZE_MAX_TOKENS`
      (2048), not the chat-sized cap, and an overrun cleans to an empty blurb —
      degraded context, never a failed file.
    - `CONTEXTUALIZE=false` keeps the identity path (`context = None`,
      `contextualized_content = content`): the measured non-contextual baseline
      and a throughput knob. Toggling it does not change content hashes.
    - The ladder: ≈35% fewer retrieval failures from contextual embeddings
      alone, ≈49% with contextual BM25 (the shipped `hybrid` default), ≈67%
      adding reranking. Measured cost on this stack: roughly 8–12 s per chunk,
      so ingest cost tracks chunk count.

### 12. What chunking strategies are registered, and what is the trap with `CHUNK_SIZE`?

??? question "Reveal solution"

    - Five, in `varagity/chunking/`: `recursive_character` (the default — 400/50,
      benchmark-kept, [ADR-008](adr/ADR-008-chunking-default.md)),
      `token_based`, `markdown_aware` (stamps a `heading_path` breadcrumb such
      as `Operations > Dredging` into the metadata), `semantic` (splits on
      embedding-similarity boundaries — 95th-percentile cosine-distance outliers
      — with the token budget only as a re-split ceiling), and
      `docling_hybrid` (Docling's structure-aware chunker; ignores overlap and
      merges peers instead).
    - **`CHUNK_SIZE`'s unit is per-strategy**: characters for
      `recursive_character` and `markdown_aware`; tokens for `token_based`,
      `docling_hybrid`, and `semantic`. Tokens are counted with tiktoken
      `cl100k_base`, a documented approximation — the real e5 count runs about
      10% higher (max +23%), which is why passages warn at ≥480 tokens against
      e5's 512 ceiling.
    - Changing the strategy or its parameters does not change content hashes,
      so the corpus goes stale until a reingest; `chunk_size`, `chunk_overlap`,
      and `chunking_strategy` are recorded on every chunk as provenance.
    - The default was decided by the fact-anchored chunker sweep, not
      preference: on the 16-chunk fixtures `hybrid` and `reranked` saturate at
      1.000 under nearly every strategy, so there was no data to justify a
      swap. `markdown_aware` is the recorded strongest candidate — a clean sweep
      at k=5 across all four methods, plus heading provenance.

## Storage

### 13. How are PostgreSQL and pgvector used?

??? question "Reveal solution"

    - As the **one** store for dense vectors and the canonical chunk metadata
      ([ADR-001](adr/ADR-001-pgvector-over-qdrant.md) — Qdrant-GPU dropped,
      FAISS never durable): `documents` (one row per source file — the
      idempotency marker plus provenance; `n_chunks = 0` for textless files)
      and `chunks` (`chunk_id` PK, `doc_id` FK with `ON DELETE CASCADE`,
      `original_index`, `content`, `context`, `contextualized_content`,
      `embedding vector(1024)`, and `metadata JSONB` holding the full
      `ChunkRecord`).
    - Search is cosine: e5 vectors are L2-normalized, so an HNSW index with
      `vector_cosine_ops`, `ORDER BY embedding <=> :qvec`, and
      `score = 1 − distance`. HNSW parameters stay at defaults until eval data
      motivates tuning.
    - Three indexes beyond the primary key: the cosine HNSW, `chunks(doc_id)`,
      and the **unique** `(doc_id, original_index)` that guards the fusion
      identity.
    - It is also the hydration source for the BM25 and hybrid arms
      (`fetch_by_identity`, an `unnest` join on the identity pairs) —
      Elasticsearch stores no metadata.
    - Since v2 the same database holds conversation persistence
      (`conversations`, `messages`, `message_sources`, `conversation_groups`)
      and `app_settings` (runtime overrides plus the `_corpus_stale` flag),
      reached through the migration runner.
    - Why: one store for vectors *and* metadata, SQL inspectability as a
      debugging feature, transactional upserts for idempotency, and both GPUs
      already spoken for — HNSW on CPU is plenty at this scale. Gotchas:
      `schema.sql` runs only on first boot, and the `pgdata` volume freezes the
      first-boot password.

### 14. What does Elasticsearch store, and what does it deliberately not store?

??? question "Reveal solution"

    - One index, `varagity_contextual_bm25` (`varagity/stores/bm25_store.py`,
      ported from the cookbook's `ElasticsearchBM25`): `content` and
      `contextualized_content` as analyzed `text` fields under the built-in
      `english` analyzer with BM25 similarity; `doc_id`, `chunk_id`, and
      `original_index` stored with `"index": false` — identity only.
    - Search is a `multi_match` over both text fields with the raw query (no e5
      formatting — BM25 is lexical). Documents are addressed by `chunk_id` as
      `_id`, so re-indexing overwrites rather than duplicates, and the
      un-indexed identity fields keep doc values, which is what lets
      reingest's term-level `delete_by_query` work.
    - It deliberately holds **no** source, page, or `ChunkRecord` metadata: the
      BM25 arm returns identity plus text only, and full rows are hydrated from
      pgvector by `(doc_id, original_index)` — pg is the single source of truth
      for metadata.
    - Operations: single-node, so `yellow` is healthy and only `red` is a
      problem; a host disk more than 90% full trips the percentage disk
      watermarks (cluster `red`, writes hang — the compose service keeps the
      defaults, the throwaway testcontainers disable the check); the client
      major must match the server major (`elasticsearch>=9,<10` against
      9.2.0); `xpack.security` is off — the dev-only posture.

### 15. How does the database schema evolve after first boot?

??? question "Reveal solution"

    - `schema.sql` is mounted into `docker-entrypoint-initdb.d/` and runs
      **only on first boot** (empty data directory) — the fresh-install fast
      path holding the v1 chunk-side tables; `docker compose down -v` resets it.
    - Everything newer is ordered, idempotent SQL in
      `varagity/stores/migrations/NNN_*.sql`, applied by a hand-rolled runner
      (`migrate.py`; [ADR-005 §6](adr/ADR-005-web-stack-and-api.md) records
      Alembic as the fallback): filename order, a file not matching
      `^\d{3}_[a-z0-9_]+\.sql$` fails loudly, applied names are tracked in
      `schema_migrations`, **one transaction per file** (a failure rolls back
      atomically and is retried next boot), convergent because every statement
      is `IF NOT EXISTS`-safe.
    - Applied by the **API's lifespan on startup** (in a threadpool), never by
      the CLI — so an existing v1 `pgdata` volume gains the v2 tables on the
      next `api` boot with no `down -v`. An unreachable postgres at startup is
      tolerated (logged; persistence routes answer 503 until it returns); an
      actual SQL failure fails startup, because serving a half-applied schema
      would be worse than not starting.
    - Current files: `001_conversations`, `002_app_settings`,
      `003_condensed_query`, `004_message_engine`, `005_conversation_groups`,
      `006_graph_turns`. The rule: keep `schema.sql` and the migrations in sync
      by hand.

### 16. How is chat history persisted so that old conversations still make sense after a reingest?

??? question "Reveal solution"

    - Single-user history lives in the same Postgres, in deliberately
      independent tables (migration 001): `conversations` (auto-titled from the
      first question), `messages` (role, content, and assistant-only
      provenance: `retrieval_method`, per-stage `latency_ms` JSONB, the captured
      `reasoning` stream, v3's `condensed_query` and `chat_engine`), and
      `message_sources` (one row per evidence chunk,
      `PRIMARY KEY (message_id, rank)`).
    - `message_sources.trace` is a **snapshot, not a join**: it persists the
      score, `content`, `context`, `source`, file name and type, `page`,
      `extraction`, the file timestamps, and the serialized `RetrievalTrace` —
      everything the evidence panel renders. `chunk_id` is kept as a
      deliberately **soft reference** (no FK to `chunks`): a hard FK would
      either cascade history away on reingest or block corpus maintenance. So a
      historical conversation still explains itself after a reingest changes
      every `chunk_id`.
    - The same snapshot discipline covers the chat engine: the engine *name* is
      persisted even when it degraded, with `condensed_query` NULL marking
      "searched verbatim" — the columns must outlive the settings that produced
      the answer.
    - A turn persists in **one transaction** at `done` (`append_message`: user
      message, assistant message, sources best-first, `updated_at` bump); an
      aborted stream persists nothing. History for the condenser loads bounded
      in SQL (`recent_turns`: role and content only, the newest
      `CONDENSE_HISTORY_TURNS`), never through the full hydrating fetch.
    - Sidebar groups (migration 005) are organization, not ownership: a nullable
      FK with `ON DELETE SET NULL`, and moving a conversation never bumps
      `updated_at`.

## Retrieval

### 17. Why is e5's input formatting asymmetric, and where is it enforced?

??? question "Reveal solution"

    - `multilingual-e5-large-instruct` expects **passages raw** (the model card:
      no instruction for retrieval documents) and **queries wrapped** as
      `Instruct: {task}\nQuery: {query}`, using the card's default retrieval
      task ("Given a web search query, retrieve relevant passages that answer
      the query"). Getting it wrong does not error — it **silently degrades
      recall**.
    - Both modes live only in `varagity/models/embeddings.py`:
      `embed_passages()` (ingest) and `embed_query()` via `format_query()`
      (retrieval). Encapsulating both in one client is the direct consequence
      of embedding-over-HTTP ([ADR-002](adr/ADR-002-infinity-over-fastembed.md)):
      no caller can format inconsistently.
    - Consequence elsewhere: HyDE embeds its hypothetical passage with
      `embed_passages` on purpose — the paper's document-encoder choice; the
      probe must land in the corpus's own vector space.
    - The client warns at ≥480 tokens because e5 truncates at 512 — the guard
      that catches bigger chunks or long blurbs at ingest time.

### 18. How does `hybrid` retrieval combine the two arms?

??? question "Reveal solution"

    ```text
    score(identity) = Σ over lists  weight_list × 1 / (rank + 1)     # rank = 0-based position in that list
    SEMANTIC_WEIGHT = 0.8   BM25_WEIGHT = 0.2                          # validated to sum to 1.0
    ```

    - Over-retrieve `k × 10` from **each** store (`OVERSAMPLE = 10`, spec
      §11.4): the semantic arm is a pgvector cosine top-k on the e5 query-mode
      vector; the BM25 arm is an Elasticsearch `multi_match` on the raw text.
    - **Weighted reciprocal-rank fusion**: only rank positions matter — the two
      stores' raw scores are never compared. An identity present in both lists
      accumulates both contributions, which is also what dedupes it (on
      `(doc_id, original_index)`). Sort, keep the top-k; the maximum fused score
      is 1.0; ties keep semantic-list order.
    - **Hydrate** full rows from pgvector (`fetch_by_identity`) — Elasticsearch
      returned identity and text only. A key indexed in ES but missing from pg
      is dropped with a warning rather than surfacing an uncitable result.
    - `fuse_with_traces` is the trace-building sibling of `fuse` (identical
      math, delegated): it keeps the per-arm rank/score maps into a
      `RetrievalTrace` per survivor, and `hydrate` attaches it with
      `final_rank == fused_rank`.
    - It is the shipped default (`RETRIEVAL_METHOD=hybrid`, the ≈49% tier) and
      the base that both `reranked` and `hyde` compose. In the flows, query
      embedding is hoisted into its own tracked `embed_query` stage through the
      `encode_query()` seam, so nothing is encoded twice.

### 19. How does `reranked` work, and why is `RERANK_ENABLED` a kill switch rather than a method?

??? question "Reveal solution"

    A **composing** retriever (`varagity/retrieval/reranked.py`,
    [ADR-006](adr/ADR-006-reranking-wired.md)) — not a flow stage, not a fork of
    fusion:

    1. over-fetch a pool of `max(RERANK_CANDIDATES, k)` (default 40 — the
       cookbook's 150→20 over-fetch scaled to this corpus) from
       `RERANK_BASE_METHOD` (default `hybrid`; `semantic`/`bm25`/`hybrid`/`hyde`,
       never `reranked`);
    2. cross-encode every candidate's **original `content`** (the blurb already
       did its job at the embedding/BM25 stage) against the query at infinity's
       `POST /v1/rerank` with `bge-reranker-v2-m3` — a dedicated `httpx`
       `RerankClient`, since the endpoint is not an OpenAI-SDK method, with the
       same `tenacity` posture as the other clients;
    3. keep `min(k, RERANK_TOP_N)` (default 5), writing `rerank_score`,
       `rerank_delta = pre − post` (+ means moved up), and `final_rank` onto
       each trace; the chunk's `score` becomes the cross-encoder relevance.

    - Re-ranking **narrows** — it never invents candidates — hence the
      validators: `0 < RERANK_TOP_N ≤ RERANK_CANDIDATES`, and with
      `RETRIEVAL_METHOD=reranked`, `RERANK_TOP_N ≤ TOP_K ≤ RERANK_CANDIDATES`.
    - `RERANK_ENABLED=false` is a **kill switch orthogonal to method
      selection**: `reranked` then passes the base ranking through the same cut
      and logs the degradation. That is what lets the GUI toggle and the eval
      baseline work without renaming the method, and the persisted
      `retrieval_method` still honestly reads `reranked`. The same shape was
      reused for `HYDE_ENABLED`, `CONDENSE_ENABLED`, `PREVIEW_ENABLED`, and
      `GRAPH_ENABLED`.
    - Cost and placement: ≈0.66 s per 40-candidate pool on the RTX 5060, timed
      on its own as `stage="rerank"` while the flow's `retrieve` observation
      deliberately includes it; no new container — the reranker rides the
      embedding container. Only a served cross-encoder is valid at `/rerank`;
      pointing `RERANK_MODEL` at a bi-encoder fails as a clear `ValueError`,
      not a retry loop. The shipped `.env` keeps `RERANK_ENABLED=false` with
      `RETRIEVAL_METHOD=hybrid`; the ≈67% tier is one flip away.

### 20. What is the `RetrievalTrace`, who fills it, and who reads it?

??? question "Reveal solution"

    - A query-time record on each `RetrievedChunk` (`varagity/stores/records.py`,
      spec_v2 §9.2) — never stored on chunk rows. Fields: `semantic_rank` /
      `semantic_score`, `bm25_rank` / `bm25_score` (`None` when that arm never
      surfaced the chunk), `fused_score` / `fused_rank` (required),
      `rerank_score` / `rerank_delta` (`None` when rerank is off the path), and
      `final_rank`. All ranks 1-based, display-ready.
    - Filled in layers: `hybrid.fuse_with_traces` builds it during fusion (raw
      arm scores ride along for display only — they never enter the fusion
      math); `hydrate` attaches it with `final_rank == fused_rank`;
      `RerankedRetriever.apply_rerank` overwrites the rerank fields and
      `final_rank` on the survivors, synthesizing a trace for any trace-less
      candidate. Single-arm retrievers (`semantic`, `bm25`) report their arm as
      the fused values — there is nothing to fuse.
    - Consumers of the *same* record: the CLI matches table at `-v 2`
      (`sem #1 · bm25 #3 · fused 0.94 · rerank +2`), the web evidence panel's
      badges, the `message_sources.trace` snapshots for history, and the
      `varagity_rerank_delta` histogram. One data model, four consumers — "the
      trace is the product as much as the ranking".
    - `RetrievedChunk.trace` defaults to `None`, so pre-trace callers (the eval
      harness, raw store results) are unaffected.

### 21. What does HyDE change in the pipeline, and how must it be paired with reranking?

??? question "Reveal solution"

    `hyde` (`varagity/retrieval/hyde.py`,
    [ADR-016](adr/ADR-016-hyde-retrieval.md)) attacks the other end from
    reranking: instead of re-ordering what dense retrieval found, it changes
    *what the dense arm searches with*. It composes a base retriever exactly as
    `reranked` does:

    1. `encode_query()` makes one non-streaming LLM call that writes a
       hypothetical answer passage (`HYDE_MAX_TOKENS=1024` — a 512 cap starved
       about a third of generations on the reasoning model), post-processed
       exactly like the condense stage: `clean_response()`, a `PASSAGE:`
       label-echo strip, then empty/overlong guards (`HYDE_MAX_CHARS=2000`);
    2. the passage is embedded in e5 **passage mode** — the corpus's own vector
       space, where passage↔passage neighbors are the chunks that *look like*
       the answer;
    3. `retrieve()` hands `HYDE_BASE_METHOD` (`semantic` | `hybrid`) the
       **original query text** plus that vector through the `query_vector` seam
       — dense-arm-only substitution, so a hybrid base's BM25 arm keeps exact
       keyword recall, traces pass through untouched, and the answer prompt
       never sees the hypothetical.

    - Stacking goes one way: `RETRIEVAL_METHOD=reranked` +
      `RERANK_BASE_METHOD=hyde` — HyDE shapes the candidate pool, the
      cross-encoder judges the user's **real** query. `HYDE_BASE_METHOD=reranked`
      is config-rejected (it would judge relevance to a guess), as are `bm25`
      (it ignores query vectors) and recursion.
    - Failure is a fallback: a raised call, an empty or overlong passage, or
      `HYDE_ENABLED=false` all degrade to the base method's raw-query retrieval
      at `WARNING`. The generation is timed as `stage="hyde"` inside the flow's
      `embed` observation, mirroring rerank inside `retrieve`.
    - Cost ~12.5 s per query on this stack, on the same llama.cpp server that
      answers; the passage is never persisted. The default stays `hybrid` —
      HyDE was added to be *evaluated* (matrix configs 6–7), and because the
      fixture corpus saturates, the verdict waits on the discriminative corpus.

## Query path, chat & API

### 22. Walk through what happens when a question is submitted from the GUI.

??? question "Reveal solution"

    1. **Request.** `POST /api/chat` with `{query, conversation_id?, overrides?,
       corpus?}`. The browser cannot use `EventSource` (GET-only), so it
       `fetch()`es and parses the body with `eventsource-parser`.
    2. **Pre-stream checks, cheapest first** (`prepare_chat`): body shape (422)
       → retrieval-method resolution (`422 unknown_retrieval_method`) →
       chat-engine resolution (`422 unknown_chat_engine`) → dependency preflight
       (`503 <service>_unreachable` for postgres, elasticsearch, llamacpp,
       infinity — prefect deliberately absent, since flows fall back to an
       ephemeral in-process API) → conversation existence (`404`), whose store
       round-trip doubles as the bounded history load. Anything detectable
       before the 200 flushes is a clean structured status.
    3. **The flow.** `query_stream_flow` runs in a worker thread, every stage a
       tracked Prefect task: `condense_query` (the chat engine returns the
       `PreparedQuery` split) → `embed_query` (`retriever.encode_query(search_query)`)
       → `retrieve` (`semantic` / `bm25` / `hybrid` / `reranked` / `hyde`) →
       `on_retrieved` fires the SSE **`retrieval`** event — chunks with traces,
       `method`, `top_k`, `reranked_to`, `condensed_query` — **before any answer
       token** → `generate_answer_stream_task` streams the grounding prompt's
       deltas while remaining a tracked task.
    4. **Streaming.** `ThinkStreamSplitter` classifies each fragment into a
       `reasoning` event (inside `<think>…</think>`) or a `token` event;
       throttled `stats` frames (live decode throughput) interleave only when
       llama.cpp reports `timings` (no frame before 8 decoded tokens, at least
       250 ms apart). `EventBridge` carries frames from the worker thread onto
       the event loop (`call_soon_threadsafe` into an `asyncio.Queue`).
    5. **Done.** The turn persists in one transaction (a new conversation is
       created if none was named), auto-titling fires and forgets, and `done`
       carries `message_id`, `conversation_id`, the full `<think>`-stripped
       answer (authoritative — the streamed deltas are best-effort display),
       and `usage` with per-stage `latency_ms`. A failure after the 200 flushed
       is an in-band `error` event (`pipeline_error`).
    6. **Render.** The web app rewrites `[SOURCE]` markers into citation chips
       *before* markdown parsing and matches them to the evidence panel, which
       was already populated from the `retrieval` frame while the answer
       streamed.

### 23. What does "async at the edge, sync flows underneath" mean, and why was it chosen?

??? question "Reveal solution"

    - FastAPI handlers are `async`, but they run the **unchanged synchronous**
      Prefect flows in a worker threadpool (`run_in_threadpool`) and marshal the
      streaming callbacks back onto the event loop. No pipeline code was
      rewritten to async — the `openai` SDK, psycopg, the Elasticsearch client,
      and in-process Prefect are all synchronous
      ([ADR-005 §1](adr/ADR-005-web-stack-and-api.md)).
    - The bridge is `EventBridge` (`varagity/api/streaming.py`): built on the
      running loop, its `emit()` is called from the flow's thread and enqueues
      via `call_soon_threadsafe`; the async generator drains the queue in
      order; `abort()` sets a `threading.Event` the flow polls between tokens.
      Thread-safe, order-preserving, and the only place the two worlds touch.
    - Why: the CLI and the API stay peer front-ends over the same flows, so the
      API can never diverge from what the pipeline does; the event loop stays
      free to stream while a flow occupies one worker thread; and a client
      disconnect frees the GPU between tokens.
    - The cost: pipeline calls block threads, so concurrency is bounded by the
      threadpool — acceptable for the single-user posture. Related: the API
      runs a **single uvicorn worker** so process-local state (the in-memory
      ingest runner, the Prometheus registry) stays coherent; scale with
      container replicas, not `--workers`. Every error, unhandled 500s
      included, is enveloped as `{error: {code, message}}` by a pure-ASGI
      middleware seated inside CORS, so cross-origin browsers see the real
      error instead of `TypeError: Failed to fetch`.

### 24. How do chat engines decide what the retriever searches with, and what is the two-string invariant?

??? question "Reveal solution"

    - A third registry (`varagity/chat/`, `CHAT_ENGINE`,
      [ADR-011](adr/ADR-011-chat-engine-condense.md)): an engine decides *what
      string the retriever searches with*, given the turn and its history. It
      is deliberately not a `Retriever` — condensing needs history, which
      `retrieve(query, k, …)` has no slot for.
    - `simple` (the default) is the identity split: no LLM call, about 3 ms.
      `condense_context` makes one non-streaming LLM call that rewrites a
      follow-up into a standalone search query against the newest
      `CONDENSE_HISTORY_TURNS` (6) turns — LlamaIndex's Condense-Plus-Context
      pattern in registry form.
    - The **two-string split** (`PreparedQuery`): `search_query` drives both the
      query embedding and BM25; `original_query` — the user's verbatim words —
      always reaches the answer prompt. A bad rewrite may cost recall, but it
      can never misstate what was asked. The SSE `retrieval` event carries
      `condensed_query` ("Searched for: …"), and migrations 003/004 persist it
      alongside the engine name.
    - The condense stage is **always in the graph** as a tracked task
      (`NO_CACHE`, no retries), even under `simple`. Failure is a fallback: a
      raised call, an empty result, or one over `CONDENSE_MAX_CHARS` searches
      with the raw query at `WARNING`; the first turn never condenses.
      `CONDENSE_ENABLED=false` is the kill switch — the engine name is still
      persisted, with `condensed_query` NULL.
    - The trap: `LLMClient.generate()` returns reasoning tags verbatim, so the
      output **must** pass `clean_response()` plus the `STANDALONE QUERY:` label
      strip — an unstripped `<think>` block would ride straight into e5 as the
      search query. The eval also caught `CONDENSE_MAX_TOKENS=128` starving the
      reasoning model (0 of 11 follow-ups condensed); it is now 512.
    - The default stays `simple` by measurement: condensing wins pronoun
      follow-ups decisively (recall@1 1.000 vs 0.600 under `reranked`) and
      never drags a topic shift, but costs a mean 8.6 s LLM call before
      retrieval. The recorded revisit seam is `CONDENSE_MODEL_TYPE` — point the
      condenser at a small non-thinking model when one lands.

### 25. What happens when the user closes the tab mid-answer?

??? question "Reveal solution"

    - The client disconnect cancels the chat route's async generator; its
      `finally` calls `bridge.abort()`, which sets a `threading.Event`.
    - `generate_answer_stream` polls `should_abort()` **between deltas**, breaks
      out of the loop, and closes the SDK stream (`deltas.close()`), which
      aborts llama.cpp's server-side generation — the GPU is freed at the next
      token, not at the end of the answer.
    - An abort is a deliberate stop, not a failure: the task completes normally
      with `aborted=True`, the flow counts
      `varagity_query_total{outcome="aborted"}`, and the API persists
      **nothing** for the turn — the turn (both messages, evidence snapshots,
      timings) is persisted only at `done`, whose ids prove it.
    - The abandoned flow future's outcome is consumed so asyncio does not warn,
      and no Prefect retry fires because query-path tasks carry none.

### 26. How are citations produced and rendered, and what is the CommonMark landmine?

??? question "Reveal solution"

    - `format_context` renders each retrieved chunk as a provenance block —
      `[SOURCE]:  {source}` / `[CONTEXT]: {context}` / `[CONTENT]: {content}` —
      and the grounding prompt (`ANSWER_PROMPT`, spec §10.2, verbatim) says:
      answer using ONLY the context, say you don't know otherwise, cite the
      `[SOURCE]` of any facts used. `format_source_block` is the single owner
      of that format because `[SOURCE]` is a contract, not decoration.
    - The model emits `[SOURCE]: /abs/path` or `[SOURCE: /abs/path]`.
      `web/lib/citations.ts` extracts the markers, matches each cited path
      against the answer's evidence rows (exact source → `/`-separated suffix →
      basename, case-insensitive; a space-truncated capture is extended against
      the evidence), and rewrites them into `[basename](#varagity-cite-N)`
      links the renderer turns into chips. A citation that matches no evidence
      row is flagged "not in evidence" — a cheap grounding-drift signal.
    - The landmine: a **line-initial `[SOURCE]: /path` is a CommonMark
      link-reference definition** and is silently swallowed by the markdown
      renderer. The rewrite therefore runs *before* markdown parsing —
      load-bearing, not cosmetic.
    - The evidence also decides where a token ends, so a graph label with a
      closing bracket survives intact, and the chips link to the evidence
      panel's cards, which were populated from the `retrieval` frame.

### 27. How do runtime settings work, and what is the stale-corpus flag?

??? question "Reveal solution"

    - Modules read a cached `pydantic-settings` `Settings` via `get_settings()`,
      never `os.getenv`; validation fails fast (fusion weights sum to 1.0, the
      vocabularies, the rerank bounds, the context-window arithmetic).
    - `PATCH /api/settings` validates the **merged whole** through every
      validator (an invalid patch changes nothing), persists one `app_settings`
      row per overridden field in its **env-string** form, replays the rows as
      process environment variables (env beats `.env`), and clears the settings
      cache — so every `get_settings()` reader sees the override on its next
      call. Overrides are replayed at API startup after the migrations; rows
      that no longer validate are skipped with an error log so the API still
      boots on env defaults.
    - Query-time knobs (`RETRIEVAL_METHOD`, `TOP_K`, the fusion weights, the
      rerank toggles, `LLM_TEMPERATURE`, `MAX_TOKENS`, `CHAT_MODEL_TYPE`,
      `CHAT_ENGINE` and the `CONDENSE_*` family) take effect on the next
      question — no restart, no reingest. `PREVIEW_*` and `GRAPH_ENGINE` are
      env-only.
    - Ingest-time knobs (`CHUNKING_STRATEGY`, `CHUNK_SIZE`/`CHUNK_OVERLAP`,
      `CONTEXTUALIZE`, `OCR_ENGINE`) do not change content hashes, so changing
      one on a non-empty corpus sets `_corpus_stale` (keys beginning with `_`
      are reserved app metadata, never overrides) and the GUI shows "Re-ingest
      to apply". It records "the corpus may not match the current settings",
      not a diff: patching the value back does not clear it, a CLI
      `ingest --reingest` (another process) does not, a composer upload
      (`reingest=false`) does not — **only a completed API-driven
      `reingest=true` run** does.

## Observability, orchestration & evaluation

### 28. Why did the Ingestion dashboard read zero while the metrics were correct, and what fixed it?

??? question "Reveal solution"

    - Two compounding facts
      ([ADR-013](adr/ADR-013-corpus-gauges-vs-counters.md)). Metrics are
      **per-process**: Prometheus scrapes only the API, so a CLI ingest records
      into a registry that dies with its process. And a **labelled counter's
      child series is born at its full value** —
      `varagity_ingest_docs_total{file_type="md"}` does not exist until its
      first `.inc()` after a process start, so Prometheus's first sample is
      already `4`, there is no `0 → 4` rise inside the series, and
      `increase()`/`rate()` read **0 over any window**, `$__range` included.
      The unlabelled contextualize histogram is initialised at definition,
      which is why its panel worked and the proposed ratio returned `+Inf`.
    - The fix: corpus size is a **gauge question answered by the store**. A
      `CorpusCollector` queries pgvector at scrape time —
      `varagity_corpus_documents`, `_chunks`, `_documents_by_type{file_type}`,
      `_chunks_by_strategy{chunking_strategy}` (the strategy read from the
      `metadata` JSONB) — cached for 10 s as a module constant (deliberately not
      an env var), serving the last good snapshot through an outage and
      emitting **no samples** rather than zeros when the store was never
      reachable. `varagity_ingest_last_run_*` gauges answer "did the last run
      happen, when, how big". The counters stay for per-event questions, and
      the gauges see CLI ingests too.
    - Made unrepresentable: `tests/unit/test_dashboards.py` fails any panel
      using `increase()`/`rate()` over a `varagity_ingest_*` counter, and any
      expression naming an uncatalogued metric or label.
    - Two cousins recorded alongside: the prefect-exporter windows its flow-run
      queries to `OFFSET_MINUTES` (image default **3**; compose sets 1440) —
      the same bug class one layer up — and `varagity_rerank_delta` exposes no
      `_sum` because its buckets are signed, so chart it with bucket quantiles,
      never sum/count averages.

### 29. Describe the Prefect posture: how do flows run, and what caching and retry choices were made?

??? question "Reveal solution"

    - Every pipeline is a Prefect flow (`varagity/pipeline/`) whose stages are
      thin `@task` adapters over plain module functions — the ingest loop
      invokes its stages through the `IngestStages` seam, so plain and tracked
      execution share one loop and cannot drift. Flows: `ingest`, `query`,
      `query-stream`, and the `eval-*` family.
    - Flows run **in-process** from the CLI and the API — no workers,
      deployments, or schedules; the server keeps its default SQLite in the
      `prefect` volume ([ADR-003 §2](adr/ADR-003-vertical-build-and-ops-choices.md)).
      `PREFECT_API_URL` is exported **before `prefect` is imported**
      (`varagity/pipeline/__init__.py`) because Prefect captures its
      environment at import time; with no server reachable, Prefect 3 falls
      back to an ephemeral in-process API, so host runs work untracked — and
      the chat preflight deliberately excludes prefect.
    - Every task sets `cache_policy=NO_CACHE`: the stages are side-effecting
      calls with unhashable inputs (store clients, progress displays) — the
      default input-hash policy would log an error per run, and a cache hit
      could never be correct. `validate_parameters=False` because tests and the
      eval harness inject duck-typed fakes.
    - Retries in layers with different scopes: `tenacity` inside every
      model/store client retries transient HTTP *within* one call; ingest's
      `contextualize`/`embed`/`store` tasks additionally carry `retries=2` with
      backoff to re-run a whole stage (safe — both store writes are
      idempotent); discovery/parse/chunk carry none (local and deterministic);
      query-path tasks carry **none** — the path is interactive, stacked
      backoff would multiply the wait before a hard failure surfaces, and
      condense must fail fast into its raw-query fallback.
    - Flow bodies double as the Prometheus probe points (stage timings, scores,
      outcomes, tokens). Measured tracking overhead is ≈0.06 s on a ~7.5 s
      question. The three output channels stay separate: `verbose` rendering
      (`debug/show.py`), stdlib logging (configured only in
      `logging_setup.py`), and Prefect run logs.

### 30. How is retrieval quality measured, and how does the eval avoid touching the live corpus?

??? question "Reveal solution"

    - `uv run --group eval main.py eval` runs the `eval-matrix` flow over
      `varagity/eval/`: it spins **ephemeral testcontainers** Postgres and
      Elasticsearch per run
      ([ADR-003 §3](adr/ADR-003-vertical-build-and-ops-choices.md); the
      throwaway ES disables its disk-watermark checks) and uses the live GPU
      services, which are stateless — the live stores are never touched.
    - Two ingests cover the seven-configuration ladder: ingest A
      (`CONTEXTUALIZE=false`) → (1) semantic non-contextual; ingest B
      (contextual, reingest) → (2) semantic contextual, (3) contextual BM25,
      (4) hybrid contextual, (5) hybrid + rerank contextual (the ≈67% tier),
      (6) HyDE → hybrid, (7) HyDE → hybrid + rerank. Eval pins its settings
      (`recursive_character` 400/50; `RERANK_CANDIDATES=40` and
      `RERANK_TOP_N=20` so recall@10/20 stay meaningful).
    - Metrics: `recall@k` (per-query fraction of golden chunks in the top-k,
      averaged — the cookbook's headline number) and `pass@k` (share of
      queries with *all* golden chunks in the top-k) for k ∈ {5, 10, 20}. The
      golden set (`data/eval/golden_qa.jsonl`) names relevant chunks portably
      by `rel_source` + `chunk_index` + a literal `fact` snippet, so refs
      resolve to `chunk_id`s from corpus files alone.
    - The **chunker sweep** rides the same run: every registered strategy is
      ingested contextually and measured under all four methods, with golden
      refs re-resolved **by fact** (`resolve_golden_by_fact`) because foreign
      chunk boundaries make `chunk_index` meaningless; an unmatched fact is a
      guaranteed miss, never dropped. HyDE configs are excluded from the sweep —
      the passage depends on the query, not the boundaries.
    - The honest caveat: the 16-chunk fixture corpus saturates (every config
      1.000 at k ≥ 10), so the harness proves the wiring and `reranked ≥ hybrid`;
      the discriminative verdict waits on the deferred cookbook corpus (737
      chunks / 248 queries). Siblings: `eval ocr` (CER/WER and pages/s —
      [ADR-004](adr/ADR-004-ocr-engine-choice.md)) and `eval chat` (the
      multi-turn harness that decided ADR-011). Results persist as timestamped
      JSON under `data/eval/results/`.

</div>
