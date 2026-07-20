-- v3 chat-engine provenance, part 1 of 2 (spec_v3 §8) — applied by the
-- idempotent migration runner (varagity/stores/migrate.py) on API startup.
--
-- Landed inert (plan decision #13): nothing wrote the column until the
-- condense_context engine was wired through the chat route.
-- Nullable by design — NULL means "not condensed" (a first turn, the simple
-- engine, or the condense fallback), the honest representation needing no
-- default. Snapshot semantics, like message_sources.trace: the column
-- explains a historical answer, so it must outlive the settings that
-- produced it. schema.sql is deliberately untouched: the messages table
-- lives only in 001 (plan decision #11).

ALTER TABLE messages ADD COLUMN IF NOT EXISTS condensed_query TEXT;
