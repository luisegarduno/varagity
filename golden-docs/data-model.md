# Data model

Storing complete, queryable metadata for every chunk is a hard requirement
(spec §8). This page documents the schema **as built**, including the two
deliberate amendments over the spec: the relative-path `doc_id` and the unique
identity index (both ADR-003).

## Identity derivation

Implemented in `varagity/stores/records.py`:

```
content_hash = sha256(file_bytes)                                  # bytes, not parsed text
doc_id       = sha256(f"{relative_path}:{content_hash}")[:16]      # path relative to DOCS_PATH
chunk_id     = f"{doc_id}::{chunk_index}"
```

- **`content_hash` hashes raw bytes** so unchanged files are skipped *before*
  paying the parse cost, and so `doc_id` is stable across OCR engines (bytes,
  not extracted text).
- **`doc_id` hashes the relative path** (spec §8.1 said absolute). Absolute
  paths differ between host (`/home/…/docs/a.md`) and container
  (`/app/docs/a.md`) and across machines — hashing them would break
  idempotency and make golden eval sets non-portable. The absolute path is
  still recorded in `source`.
- **`(doc_id, original_index)`** is the cross-store join/fusion identity;
  `original_index` is a global monotonic counter allocated per ingest run
  (`SELECT COALESCE(MAX(original_index), -1) + 1`, then incremented
  in-process).

## Chunk metadata (`ChunkRecord`)

Every chunk persists this record (pydantic-validated; stored whole in the
`chunks.metadata` JSONB column and mirrored into typed columns where queried):

| Field | Type | Description |
|---|---|---|
| `doc_id` | str | Stable per-document id (see above) |
| `chunk_id` | str | `f"{doc_id}::{chunk_index}"` — pg primary key and ES `_id` |
| `original_index` | int | Global monotonic chunk index (fusion key) |
| `chunk_index` | int | Chunk position within its document |
| `source` | str | Absolute file path (provenance only — never identity) |
| `file_name` | str | Basename |
| `file_type` | str | `pdf` / `txt` / `md` |
| `page` | int? | First page that contributed text (PDF; `None` otherwise)¹ |
| `content` | str | **Original** chunk text |
| `context` | str? | LLM situating blurb (`None` when ingested with `CONTEXTUALIZE=false`) |
| `contextualized_content` | str | `context + "\n\n" + content` — the text actually embedded & BM25-indexed; equals `content` when `context` is `None` |
| `chunk_size` / `chunk_overlap` | int | Parameters used, **in characters** (provenance) |
| `chunking_strategy` | str | e.g. `recursive_character` |
| `embedding_model` | str | Served model name, e.g. `infloat/multilingual-e5-large-instruct` |
| `n_tokens` | int | Approximate token count of `content` (cl100k — a documented approximation; the e5 tokenizer differs) |
| `content_hash` | str | The parent document's byte hash |
| `created_at` | datetime | Ingestion timestamp (UTC) |
| `extraction` | str | `"text"` or `"ocr_fallback"` — **beyond spec §8.1**: extraction provenance for retrieval-quality debugging (OCR noise hits BM25 keyword matching hardest) |

¹ `page` is document-level (the first page that contributed text), not
per-chunk: the shared chunker copies one `source_meta` per document, so
per-chunk page attribution has no data path yet. A future chunker with
`start_index` support plus a loader page-map lookup would make it per-chunk.

## PostgreSQL schema

`varagity/stores/schema.sql`, mounted into the postgres container's
`/docker-entrypoint-initdb.d/` — it runs **only on first boot** (empty data
directory). Reset with `docker compose down -v` (see the
[runbook](runbook.md#volumes-and-resets)).

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

-- (doc_id, original_index) is the fusion/join identity across pgvector and
-- Elasticsearch; enforcing uniqueness catches ingest bugs early.
CREATE UNIQUE INDEX IF NOT EXISTS chunks_doc_orig_uidx ON chunks(doc_id, original_index);
```

Notes:

- **Cosine, not L2**: e5 embeddings are L2-normalized, so search orders by
  `embedding <=> :qvec` (cosine distance) and reports
  `score = 1 - distance` (spec §11.2).
- **The unique identity index** is the one schema addition over spec §8.2
  (ADR-003): a duplicated `(doc_id, original_index)` would silently corrupt
  hybrid fusion, so it fails loudly at write time instead.
- **`n_chunks = 0` documents are deliberate**: a file with no extractable text
  still gets a `documents` row, so it is visibly "seen", cheaply re-warned on
  later runs without re-parsing, and re-attempted under `--reingest`. Files
  are never silently dropped.
- **Writes are upserts**: `ON CONFLICT (chunk_id) DO UPDATE` for chunks; the
  per-document write (`documents` row + all its chunks) is one transaction, so
  a partial failure leaves no idempotency marker and the next run re-attempts
  the file.

## Elasticsearch index

`varagity/stores/bm25_store.py` (default index name
`varagity_contextual_bm25`). Mirrors the Anthropic cookbook's
`ElasticsearchBM25`: analyzed text fields under the built-in `english`
analyzer with BM25 similarity; identity fields stored but **not indexed**.

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

Notes:

- **The index is contextual from its first document**: chunks are
  contextualized before they reach either store, and search is a
  `multi_match` over `content` **and** `contextualized_content`.
- **ES stores identity + text only.** Full records (source, page, context,
  metadata) are hydrated from pgvector by `(doc_id, original_index)` — pg is
  the single source of truth for metadata.
- **`"index": false` fields keep doc values**, so the term-level
  `delete_by_query` used by `--reingest` still works (a slower doc-values
  scan — fine at dev scale).
- **Documents are addressed by `chunk_id`** (`_id`), so re-indexing overwrites
  instead of duplicating — the sparse-side counterpart of the pg upsert.

## Idempotency & re-ingestion semantics

- A file whose `(doc_id, content_hash)` already exists in `documents` is
  skipped **before parsing**.
- Pipeline-setting changes (`CONTEXTUALIZE`, chunk params, `OCR_ENGINE`) do
  **not** change content hashes → unchanged files stay skipped. Re-process
  with `main.py ingest --reingest`, which deletes each discovered document
  from **both** stores (ES `delete_by_query` first, then the pg cascade
  delete) before ingesting fresh.
- The dual-write order is BM25 **first**, pgvector **last**: the pg
  `documents` row is the idempotency marker, so a failure between the two
  writes leaves no marker and the next run re-attempts the file (the
  `chunk_id`-addressed ES bulk then overwrites rather than duplicates).
- Removing a file from `docs/` does **not** remove its chunks in v1 (no
  corpus GC beyond `--reingest`).

## Golden eval dataset

`data/eval/golden_qa.jsonl` — one JSON object per line:

```json
{"query": "…", "relevant": [{"rel_source": "aurora_station.md", "chunk_index": 1}]}
```

Relevant chunks are identified **portably** by `rel_source` (path relative to
the corpus root — exactly the string `doc_id` hashes) plus `chunk_index`, so
entries resolve to concrete `chunk_id`s from corpus files alone, on any
machine. `chunk_index` values assume the pinned eval chunk boundaries
(`recursive_character`, 400/50 — `PINNED_EVAL_SETTINGS`); re-author the golden
set if those pins change. Ground-truth transcriptions of the OCR fixture PDFs
live in `data/eval/ocr_truth/`.
