-- Graph-targeted chat turns (spec_graphrag §4.2, stage-2 decision #12) —
-- applied by the idempotent migration runner (varagity/stores/migrate.py) on
-- API startup.
--
-- Both nullable, both snapshot semantics (the message_sources.trace rule):
-- they explain a historical answer, so they must outlive the settings and the
-- graph that produced them.
--
--   target_corpus  — which corpus the turn ASKED for ('rag' | 'graph'), so a
--                    turn that degraded to chunk RAG (kill switch off, engine
--                    unavailable) is still attributable. NULL = pre-stage-2
--                    history, read as chunk RAG.
--   graph_evidence — the graph retrieval's entities/relations plus the engine
--                    query mode. NULL on a degraded turn, which is exactly
--                    ADR-017's honest record: the graph produced nothing.
--
-- The cited transcript days ride in message_sources instead (chunk_id =
-- doc_key, trace.kind = 'graph_transcript'), reusing the soft-reference
-- snapshot shape rather than inventing a second evidence table.
--
-- schema.sql is deliberately untouched: the messages table lives only in
-- migration 001 (the 003/004 precedent).

ALTER TABLE messages ADD COLUMN IF NOT EXISTS target_corpus TEXT;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS graph_evidence JSONB;
