-- Varagity database schema (spec §8.2).
--
-- Mounted into the postgres container at /docker-entrypoint-initdb.d/, so it
-- runs ONLY on first boot (empty data directory). To re-run after a change:
-- `docker compose down -v` (drops the pgdata volume), then `docker compose up`.

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
