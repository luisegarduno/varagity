# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Varagity is a full-stack RAG application implementing Anthropic-style **Contextual Retrieval**
(contextual embeddings + contextual BM25 + hybrid rank fusion), self-hosted on local GPUs.
The v1 system is complete: terminal Q&A over a `docs/` corpus (PDF/txt/md with automatic
OCR fallback), grounded and cited answers, every pipeline stage a tracked Prefect task.

Where things live:
- `spec.md` — the forward-looking v1 design (§ references in docstrings point here).
- `golden-docs/` — **as-built** documentation, rendered by MkDocs (`uv run mkdocs serve`):
  architecture, data model, pipelines, runbook, ADRs, API reference.
- `thoughts/shared/plans/` — the vertically-sliced implementation plan with per-phase notes.
- `docs/` — ⚠️ the **gitignored ingest corpus** (RAG input), *not* documentation.

## Architecture

Six compose services on `varagity-net`; the Python app is a client of all of them:

1. **llamacpp** (`:8080/v1`, GPU 0) — chat LLM for answers + contextualization blurbs;
   model `.gguf` bind-mounted from `${models_volume}`.
2. **infinity-embeddings** (`:8081/v1`, GPU 1) — `multilingual-e5-large-instruct`
   (1024-dim) **and** `bge-reranker-v2-m3` at `/v1/rerank` (served but not wired into
   the v1 query path; `RERANK_ENABLED=false` is staged config). Host port binding is
   interface-specific (`192.168.86.21:8081`).
3. **postgres** (`:5432`) — pgvector; the canonical chunk metadata + dense vectors.
   `varagity/stores/schema.sql` runs on first boot only (`docker compose down -v` resets).
4. **elasticsearch** (`:9200`) — contextual BM25 index. Single-node ⇒ cluster health
   `yellow` is healthy.
5. **prefect** (`:4200`) — flow/task tracking UI; SQLite backing store.
6. **app** — the CLI (`main.py`), built from the local Dockerfile with `uv`.

Key invariant: chunks live in **both** stores, joined by `(doc_id, original_index)`;
`doc_id` hashes the path **relative to `DOCS_PATH`** + the file's byte hash. Hybrid
retrieval fuses ranked lists from both stores and hydrates full rows from pgvector.

## Commands

```bash
docker compose up -d --wait        # all six services, healthcheck-gated
bash scripts/smoke.sh              # sequenced infra checks

uv run main.py ingest              # ingest DOCS_PATH into both stores
uv run main.py ingest --reingest   # re-process (config changes don't change content hashes)
uv run main.py chat                # ingest, then terminal Q&A (default command; :quit exits)
uv run --group eval main.py eval       # 4-config retrieval matrix (needs Docker + live GPU services)
uv run --group eval main.py eval ocr   # OCR engine benchmark

uv run pytest                      # unit suite (default; coverage floor 80%)
uv run pytest -m integration       # real Postgres/ES via testcontainers (needs Docker)
uv run pytest -m e2e               # full pipeline over tests/fixtures/corpus (needs Docker)
uv run ruff check . && uv run ruff format --check .
uv run mypy varagity
uv run pre-commit run --all-files
uv run mkdocs build --strict       # docs must build clean (CI-gated)
```

Host-mode runs against the compose services need localhost env overrides
(`BASE_MODEL_API_URL`, `POSTGRES_HOST`, `ELASTICSEARCH_URL`, `PREFECT_API_URL`, and
`EMBEDDING_API_URL` via `docker compose port infinity-embeddings 8081`) — the checked-in
`.env` holds the in-container values. See `golden-docs/runbook.md`.

## Conventions (enforced, not aspirational)

- **Registries for pluggable families** (spec §5.1): parsers, chunking strategies, and
  retrievers self-register via `@register("name")` in their package; adding an
  implementation = one new file + its import line in the package `__init__`, zero caller
  edits. OCR engines use the same shape as a factory in `parsers/pdf.py`.
- **Configuration**: modules read the `Settings` object from `varagity/config.py`
  (`get_settings()`, cached) — never `os.getenv`. `.env` is consumed by both compose
  (lowercase interpolation vars) and pydantic-settings.
- **Three output channels** (spec §14): `verbose: int` (0/1/2, validated via
  `check_verbose`, rendering only in `varagity/debug/show.py` as `v_<name>` helpers);
  stdlib `logging` (configured only in `logging_setup.py`); Prefect run logs.
- **Docstrings**: Google-style on every public module/class/function — ruff's `D` rules
  fail commits without them; mkdocstrings renders them into the docs site. Args/Returns/
  Raises must match the signature.
- **e5 formatting is asymmetric** and silently degrades recall if wrong: passages get NO
  prefix, queries get `Instruct: {task}\nQuery: {q}`. Both modes live only in
  `varagity/models/embeddings.py`.
- **Retries**: `tenacity` inside model/store clients (transient HTTP); ingest model/store
  Prefect tasks additionally carry `retries=2`; query-path tasks deliberately carry none.
- **Prefect**: flows run in-process from the CLI (no workers/deployments);
  `PREFECT_API_URL` must be exported **before** `prefect` is imported
  (`varagity/pipeline/__init__.py` handles this); every task sets `cache_policy=NO_CACHE`.
- **Testing layers**: unit (default, mocked HTTP via `respx`), `-m integration`
  (testcontainers), `-m e2e` (fake embeddings/LLM + real containerized stores). Shared
  container setup lives in `varagity/eval/containers.py`.

## Gotchas worth knowing

- `pgdata` keeps the **first-boot** postgres password; editing `.env` later breaks host
  TCP auth while `compose exec psql` still works (`ALTER USER` or `down -v` to fix).
- Host disk >90% full trips Elasticsearch's percentage disk watermarks → cluster `red`,
  writes hang. Testcontainers disable the check; the compose service keeps defaults.
- infinity's `optimum` engine ignores `INFINITY_DEVICE_ID` — GPU pinning happens via
  compose `device_ids`. The reranker needs pre-exported ONNX and the `'32;4'` batch cap
  (torch has no sm_120 kernels; 8 GB card).
- llama.cpp `/health` returns 503 while loading (~30 s); healthcheck retries cover it.
  Slow prompt-eval relative to decode is the MoE `-ot` CPU-offload signature, not a bug.
- Docling/EasyOCR/tiktoken download models on first use (cached in the `model_cache`
  volume in-container).
- Toggling `CONTEXTUALIZE`/chunk params does **not** change content hashes — unchanged
  files are skipped until `ingest --reingest`.

## Package Management

`uv` for everything: `uv sync` (dependency groups: `dev`, `eval`), `uv run <cmd>`.
Add dependencies in `pyproject.toml`, then `uv sync`. Torch is pinned to CPU wheels
via `[tool.uv.sources]` (v1 OCR is CPU-only by design; saves ~3 GB).
