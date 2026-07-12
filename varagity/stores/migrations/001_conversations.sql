-- v2 conversation persistence (spec_v2 §9.1) — applied by the idempotent
-- migration runner (varagity/stores/migrate.py) on API startup, so existing
-- pgdata volumes gain the tables without `docker compose down -v`.
--
-- Independent of the RAG chunk tables by design: message_sources.chunk_id is
-- a soft reference (no FK to chunks) and `trace` snapshots the evidence, so a
-- historical conversation still explains itself after a reingest changes
-- chunk ids.

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id  TEXT PRIMARY KEY,        -- app-generated id
    title            TEXT NOT NULL,           -- auto-titled from first question
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS messages (
    message_id       TEXT PRIMARY KEY,
    conversation_id  TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    role             TEXT NOT NULL,           -- 'user' | 'assistant'
    content          TEXT NOT NULL,           -- question or generated answer
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- assistant-only provenance snapshot (null for user turns):
    retrieval_method TEXT,
    latency_ms       JSONB,                   -- per-stage timings
    reasoning        TEXT                     -- captured <think> stream, if any
);

CREATE TABLE IF NOT EXISTS message_sources (
    message_id       TEXT NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE,
    rank             INT  NOT NULL,           -- final rank in the answer's evidence
    chunk_id         TEXT NOT NULL,           -- soft ref (survives reingest as a snapshot)
    trace            JSONB NOT NULL,          -- RetrievalTrace (§9.2) + score + content/context/source snapshot
    PRIMARY KEY (message_id, rank)
);

-- Transcript fetches read a conversation's messages in order.
CREATE INDEX IF NOT EXISTS messages_conversation_created_idx
    ON messages(conversation_id, created_at);
