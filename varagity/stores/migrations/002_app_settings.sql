-- v2 runtime settings overrides (spec_v2 §4.7) — applied by the idempotent
-- migration runner (varagity/stores/migrate.py) on API startup.
--
-- One row per overridden setting, keyed by the Settings field name (e.g.
-- 'RETRIEVAL_METHOD') with the value in its env-string form — the same form
-- pydantic-settings parses from the environment, so the override layer
-- (varagity/api/runtime_settings.py) can replay rows as env vars verbatim.
-- Keys starting with '_' are reserved for app metadata (e.g. '_corpus_stale',
-- the "settings changed since the last reingest" flag behind the GUI's
-- "Re-ingest to apply" affordance).

CREATE TABLE IF NOT EXISTS app_settings (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
