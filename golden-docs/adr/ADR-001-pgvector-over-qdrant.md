# ADR-001: PostgreSQL + pgvector over Qdrant-GPU

**Status:** Accepted (spec §21 #1)

## Context

The original stack sketch listed **Qdrant-GPU** as the dense vector store,
and the reference implementation this project pattern-matches against used
**FAISS in-process**. v1 needs a durable store for ~thousands of chunks with
full metadata per chunk, cosine top-k search, cross-store identity joins for
hybrid fusion, and easy inspection while the retrieval pipeline is being
debugged and evaluated.

## Decision

Use **PostgreSQL 16 + pgvector** (`pgvector/pgvector:pg16`) as the only
dense vector store. Qdrant-GPU is dropped from the plan — not abstracted
for, dropped.

## Rationale

- **One store for vectors *and* the canonical metadata.** The hard
  requirement is a complete, queryable metadata record per chunk; pg holds
  the `ChunkRecord` JSONB, the typed columns, and the vector in one row.
  The hybrid/BM25 retrievers hydrate full records from pg by
  `(doc_id, original_index)` — that join is plain SQL.
- **SQL inspectability is a debugging feature.** Verification at every step
  ran psql queries against live rows (`context IS NULL` invariants, chunk
  counts, `extraction` provenance). A dedicated vector DB makes each of
  those a bespoke API call.
- **Durability and idempotency need a transactional store.** Per-document
  writes are one transaction; unique indexes enforce the fusion identity;
  `ON CONFLICT` upserts give idempotent re-ingest. FAISS (in-memory) has
  none of this; Qdrant has its own transactionality but adds a second data
  system to operate.
- **GPU budget is spoken for.** Both GPUs serve models (LLM + embeddings).
  A GPU-accelerated ANN index solves a scale problem v1 does not have —
  HNSW on CPU is more than adequate at this corpus size.
- **e5 embeddings are L2-normalized**, so cosine (`vector_cosine_ops`,
  `1 - (embedding <=> q)`) is the right metric and pgvector's HNSW supports
  it directly.

## Consequences

- The vector store is `varagity/stores/vector_store.py` over psycopg;
  `schema.sql` initializes the database on first boot.
- HNSW parameters (`m`, `ef_construction`, `ef_search`) stay at defaults
  until eval data motivates tuning.
- If corpus scale ever outgrows pg, the seam is the `ContextualVectorDB`
  class — but per spec §20, alternative backends are only revisited on a
  concrete need, and the eval harness exists to demonstrate one.
