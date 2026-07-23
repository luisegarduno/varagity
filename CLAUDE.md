# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Varagity is a full-stack RAG application implementing Anthropic-style **Contextual Retrieval**
(contextual embeddings + contextual BM25 + hybrid rank fusion + cross-encoder reranking),
self-hosted on local GPUs. v1 (complete) is the terminal system: Q&A over a `docs/` corpus,
grounded and cited answers, every pipeline stage a tracked Prefect task. v2 (complete)
adds: reranking wired into the query path with a per-chunk
`RetrievalTrace` (the ≈67% tier), a FastAPI SSE backend with conversation persistence +
migrations, a Next.js chat GUI whose evidence panel shows "how this answer was built" with
inline citations, office/web document modalities, four new chunking strategies (five total)
with a benchmark sweep, Prometheus/Grafana observability, corpus management + a live
settings UI, the design system (a11y, ⌘K palette, opt-in Playwright e2e), and hardening
(ADR-005…009, two-job CI with coverage floors, as-built docs refresh). v3 (complete) is
the polish release: a **chat-engine registry** whose `condense_context` engine rewrites
follow-ups into standalone search queries against conversation history (the shipped
default stays `simple` — benchmark-decided, ADR-011; the original words always drive the
answer prompt), composer 📎 uploads of files *and folders* with auto-ingest + client-side
409 queueing (ADR-012), the observability repair (store-derived corpus gauges ADR-013,
honest Infra empty-states, the prefect-exporter `OFFSET_MINUTES` fix), and pnpm→bun as
package manager only (ADR-014). Page previews shipped between v2 and v3 (ADR-010).

Where things live:
- `spec.md` / `spec_v2.md` / `spec_v3.md` — the v1/v2/v3 designs (§ references in
  docstrings point at them), untracked under `thoughts/shared/specs/`.
- `golden-docs/` — **as-built** documentation, rendered by MkDocs (`uv run mkdocs serve`):
  architecture, data model, pipelines, runbook, ADRs 001–014, Python API reference;
  `golden-docs/api.md` is the HTTP contract + SSE protocols, rendered from the
  `golden-docs/openapi.json` snapshot (regenerate via `scripts/export_openapi.py` —
  a unit test fails on drift).
- `thoughts/shared/plans/` — the vertically-sliced implementation plans with per-phase notes.
- `web/` — the Next.js frontend (own toolchain: bun, Vitest; heed `web/AGENTS.md` — the
  Next.js version post-dates training data, read `node_modules/next/dist/docs/` first).
- `docs/` — ⚠️ the **gitignored ingest corpus** (RAG input), *not* documentation.

## Architecture

Ten compose services on `varagity-net` (+ two optional exporter profiles); the Python
package is a client of all the backing services, with two peer front-ends (CLI and API)
over the same Prefect flows:

1. **llamacpp** (`:8080/v1`, GPU 0) — chat LLM for answers + contextualization blurbs;
   model `.gguf` bind-mounted from `${models_volume}`.
2. **infinity-embeddings** (`:8081/v1`, GPU 1) — `multilingual-e5-large-instruct`
   (1024-dim) **and** `bge-reranker-v2-m3` at `/v1/rerank`, wired into the query path
   (`RETRIEVAL_METHOD=reranked`). Host port binding is interface-specific
   (`192.168.86.21:8081`).
3. **postgres** (`:5432`) — pgvector; canonical chunk metadata + dense vectors, plus
   conversation persistence (`conversations`/`messages`/`message_sources`, plus
   `conversation_groups` — the sidebar's folders).
   `varagity/stores/schema.sql` runs on first boot only (`docker compose down -v` resets);
   `varagity/stores/migrations/*.sql` reconcile existing volumes (idempotent runner on API
   startup).
4. **elasticsearch** (`:9200`) — contextual BM25 index. Single-node ⇒ cluster health
   `yellow` is healthy.
5. **prefect** (`:4200`) — flow/task tracking UI; SQLite backing store.
6. **api** (`:8000`) — FastAPI (`varagity/api/`, built from `Dockerfile.api`): SSE chat
   streaming, conversation CRUD, corpus upload/ingest + runtime settings routes,
   health/config, `/metrics`; invokes the flows in-process.
7. **web** (`:3000`) — the Next.js GUI (`web/`); the only browser-facing surface, talks
   only to `api`.
8. **prometheus** (`:9090`) — scrapes `api:8000/metrics` every 15 s; optional extra
   targets via `--profile prefect-exporter` and `--profile gpu-metrics` (dcgm-exporter).
9. **grafana** (`:3001`) — provisioned Prometheus datasource + Query/Ingestion/Infra
   dashboards; anonymous viewer, default `admin/admin` for edits.
10. **app** — the CLI (`main.py`), built from the local Dockerfile with `uv`.

Key invariants: chunks live in **both** stores, joined by `(doc_id, original_index)`;
`doc_id` hashes the path **relative to `DOCS_PATH`** + the file's byte hash. Hybrid
retrieval fuses ranked lists from both stores and hydrates full rows from pgvector;
`reranked` **composes** a base retriever (over-fetch `RERANK_CANDIDATES`, cross-encode,
keep `RERANK_TOP_N`) rather than forking fusion. Each `RetrievedChunk` carries an optional
`RetrievalTrace` (per-arm ranks, fused score, rerank delta); the CLI matches table, the web
evidence panel, and the `message_sources.trace` snapshots all render that same data.

## Commands

```bash
docker compose up -d --wait        # all ten default services, healthcheck-gated
bash scripts/smoke.sh              # sequenced infra checks (all ten default services)

uv run main.py ingest              # ingest DOCS_PATH into both stores
uv run main.py ingest --reingest   # re-process (config changes don't change content hashes)
uv run main.py chat                # ingest, then terminal Q&A (default command; :quit exits)
uv run uvicorn varagity.api.main:create_app --factory --port 8000   # API on the host
uv run --group eval main.py eval       # 7-config retrieval matrix + chunker sweep (needs Docker + live GPU services)
uv run --group eval main.py eval ocr   # OCR engine benchmark
uv run --group eval main.py eval chat  # multi-turn chat-engine eval (the ADR-011 decision harness)

uv run pytest                      # unit suite incl. async API tests (coverage floor 90%)
uv run pytest -m integration       # real Postgres/ES via testcontainers (needs Docker)
uv run pytest -m e2e               # full pipeline over tests/fixtures/corpus (needs Docker)
uv run ruff check . && uv run ruff format --check .
uv run mypy varagity
uv run pre-commit run --all-files
uv run mkdocs build --strict       # docs must build clean (CI-gated)
uv run python scripts/export_openapi.py   # refresh golden-docs/openapi.json after API surface changes

# web/ (frontend — bun, not uv)
bun run dev                        # dev server against NEXT_PUBLIC_API_URL
bun run test                       # Vitest unit tests — the suite that CI coverage-gates
bun run e2e                        # opt-in Playwright — needs the live stack on :3000/:8000
bun run lint && bun run typecheck && bun run build
bun run gen:types                  # regenerate lib/types.ts from the API's OpenAPI schema
```

Host-mode runs against the compose services need localhost env overrides
(`BASE_MODEL_API_URL`, `POSTGRES_HOST`, `ELASTICSEARCH_URL`, `PREFECT_API_URL`, and
`EMBEDDING_API_URL`/`RERANK_API_URL` via `docker compose port infinity-embeddings 8081`) —
the checked-in `.env` holds the in-container values. See `golden-docs/runbook.md`.

## Conventions (enforced, not aspirational)

- **Registries for pluggable families** (spec §5.1): parsers (`pdf`, `text`, `office`,
  `web`, `image`), chunking strategies (`recursive_character`, `token_based`, `markdown_aware`,
  `semantic`, `docling_hybrid`), retrievers (`semantic`, `bm25`, `hybrid`, `reranked`, `hyde`),
  and chat engines (`simple`, `condense_context`) self-register via `@register("name")`
  in their package; adding an implementation = one new file + its import line in the
  package `__init__`, zero caller edits. OCR engines use the same shape as a factory in
  `parsers/pdf.py`. Registry vocabularies are **hardcoded again** in their `config.py`
  validators (circular import), each with a tuple↔registry regression test.
- **Configuration**: modules read the `Settings` object from `varagity/config.py`
  (`get_settings()`, cached) — never `os.getenv`. `.env` is consumed by both compose
  (lowercase interpolation vars) and pydantic-settings.
- **API layer** (spec_v2 §4): async at the edge, sync flows underneath — FastAPI runs the
  flows in a threadpool; don't rewrite pipeline code to async. `api/schemas.py` is the wire
  contract; the SSE protocol is `retrieval → reasoning → token → done` (or `error`),
  evidence before prose. Frontend types are generated (`bun run gen:types`), never hand-edited;
  the `golden-docs/openapi.json` snapshot is drift-guarded — rerun
  `scripts/export_openapi.py` after surface changes (a unit test fails otherwise).
- **No `useEffect` in `web/`** (`.claude/skills/no-use-effect`, enforced by
  `no-restricted-syntax` — `bun run lint` fails on a direct call): derive state inline,
  act in event handlers, fetch with `useQuery` over the `lib/queries.ts` factories, or
  `key` a component to re-run on a value change. The only two sanctioned `useEffect`s are
  the named primitives in `web/hooks/` (`useMountEffect` for mount-scoped external sync,
  `useDebouncedValue` for the streaming anti-flash timer), each with a scoped disable.
- **Server state lives in TanStack Query** (`web/lib/queries.ts`): one `queryOptions`
  factory per dataset, so every consumer of the same data shares a cache entry and one
  in-flight request. The window buses (`lib/*-bus.ts`) stayed the decoupling seam for
  mutating surfaces, but only `QueryBusBridge` subscribes — it turns each event into
  `invalidateQueries`. Conversation *list* and *transcript* keys are deliberately
  disjoint (invalidation is prefix-matched), so a persisted turn re-orders the list
  without discarding the transcript it was just folded into.
- **Migrations**: ordered, idempotent SQL in `varagity/stores/migrations/NNN_*.sql`,
  tracked in `schema_migrations`, applied by the API on startup. `schema.sql` stays the
  fresh-install fast path — keep both in sync.
- **Three output channels** (spec §14): `verbose: int` (0/1/2, validated via
  `check_verbose`, rendering only in `varagity/debug/show.py` as `v_<name>` helpers);
  stdlib `logging` (configured only in `logging_setup.py`); Prefect run logs.
- **Docstrings**: Google-style on every public module/class/function — ruff's `D` rules
  fail commits without them; mkdocstrings renders them into the docs site. Args/Returns/
  Raises must match the signature.
- **e5 formatting is asymmetric** and silently degrades recall if wrong: passages get NO
  prefix, queries get `Instruct: {task}\nQuery: {q}`. Both modes live only in
  `varagity/models/embeddings.py`.
- **Retries**: `tenacity` inside model/store clients (transient HTTP; the rerank client
  included); ingest model/store Prefect tasks additionally carry `retries=2`; query-path
  tasks deliberately carry none.
- **Prefect**: flows run in-process from the CLI and the API (no workers/deployments);
  `PREFECT_API_URL` must be exported **before** `prefect` is imported
  (`varagity/pipeline/__init__.py` handles this); every task sets `cache_policy=NO_CACHE`.
- **Testing layers**: unit (default, mocked HTTP via `respx`; API routes via
  `httpx.AsyncClient` + pytest-asyncio, SSE helpers in `tests/sse.py`), `-m integration`
  (testcontainers), `-m e2e` (fake embeddings/LLM + real containerized stores); web unit
  tests via Vitest (`bun run test`, runs in CI under a coverage floor); Playwright e2e is
  opt-in (`bun --cwd web run e2e`, needs the live stack). CI = two jobs per push: Python
  (ruff, mypy, unit suite at the 90% floor, `mkdocs build --strict`) + web (lint,
  coverage-gated Vitest, build); integration/e2e/Playwright stay local. Shared container
  setup lives in `varagity/eval/containers.py`.

## Gotchas worth knowing

- `pgdata` keeps the **first-boot** postgres password; editing `.env` later breaks host
  TCP auth while `compose exec psql` still works (`ALTER USER` or `down -v` to fix).
- `NEXT_PUBLIC_API_URL` is a **build-time** constant (compose build arg): changing it
  requires `docker compose build web`. Browsers on other machines need the host's LAN
  address there *and* in `API_CORS_ORIGINS`.
- `CHUNK_SIZE`'s unit is **per-strategy**: characters for `recursive_character` /
  `markdown_aware`, tokens for `token_based` / `docling_hybrid` (`semantic` splits on
  embedding-similarity boundaries instead).
- `RERANK_ENABLED=false` is a kill switch, not a method: `RETRIEVAL_METHOD=reranked` then
  degrades to its base method (and logs it). Method selection and the toggle are
  deliberately orthogonal. `HYDE_ENABLED=false` is the same shape for `hyde`
  (degrades to `HYDE_BASE_METHOD`); pair HyDE with reranking as
  `RETRIEVAL_METHOD=reranked` + `RERANK_BASE_METHOD=hyde` — never hyde-over-reranked
  (config-rejected: the cross-encoder must judge the real query).
- `CONDENSE_ENABLED=false` is likewise a kill switch, not an engine:
  `CHAT_ENGINE=condense_context` then degrades to `simple` behavior (and logs it).
  Engine selection and the toggle are deliberately orthogonal — and the engine name is
  still persisted on the turn, with `condensed_query` NULL marking the degrade.
- The condenser's output **must** pass through `clean_response()` (plus the
  `CONDENSE_QUERY_LABEL` echo strip): llama.cpp emits `<think>` blocks and the
  non-streaming `LLMClient.generate()` does **not** strip them — an unstripped one goes
  straight into the embedding model as the search query and silently destroys retrieval.
  The HyDE passage has the identical trap (`clean_response()` + `HYDE_PASSAGE_LABEL`
  strip in `retrieval/hyde.py`).
- Line-initial `[SOURCE]: …` is a CommonMark link-reference *definition* and silently
  vanishes when rendered — the web app rewrites citations to chips **before** markdown
  parsing (`web/lib/citations.ts`). Mind this when touching answer rendering.
- `message_sources.trace` **snapshots** evidence (content/context/source + trace) so
  historical conversations still explain themselves after a reingest changes `chunk_id`s —
  the chunk reference is deliberately soft (no FK).
- Host disk >90% full trips Elasticsearch's percentage disk watermarks → cluster `red`,
  writes hang. Testcontainers disable the check; the compose service keeps defaults.
- infinity's `optimum` engine ignores `INFINITY_DEVICE_ID` — GPU pinning happens via
  compose `device_ids`. The reranker needs pre-exported ONNX and the `'32;4'` batch cap
  (torch has no sm_120 kernels; 8 GB card).
- llama.cpp `/health` returns 503 while loading (~30 s); healthcheck retries cover it.
  Slow prompt-eval relative to decode is the MoE `-ot` CPU-offload signature, not a bug.
- Docling/EasyOCR/tiktoken download models on first use (cached in the `model_cache`
  volume in-container).
- Toggling `CONTEXTUALIZE`/`CHUNKING_STRATEGY`/chunk params does **not** change content
  hashes — unchanged files are skipped until `ingest --reingest`.
- Metrics are **per-process**: CLI ingests record into the CLI's own (never-scraped)
  registry and never reach Grafana — run ingests through the API/GUI to populate the
  Ingestion dashboard. (The `varagity_corpus_*` gauges are the exception: they read
  pgvector at scrape time, so they see CLI ingests too — ADR-013.)
- `increase()`/`rate()` over the ingest counters returns **0 over any window**: a
  labelled counter's child series is born at its full value, so Prometheus never sees the
  rise. Corpus size is a gauge question (`varagity_corpus_*`); counters only answer
  per-event questions — `tests/unit/test_dashboards.py` fails any panel that regresses.
- `prefecthq/prometheus-prefect-exporter` windows flow-run metrics to the last
  `OFFSET_MINUTES` (image default **3**) — a bursty workload reads 0; compose sets 1440.
  And `PREFECT_API_URL` **must** keep its `/api` suffix: the exporter's healthz appends
  `/health` and `SystemExit`s, so a wrong URL is a crash-loop, not an empty panel.
- The stale-corpus flag is cleared only by a **completed API-driven** `reingest=true` run —
  not by CLI `ingest --reingest`, not by patching the setting back, and not by a composer
  📎 upload (those auto-ingest with `reingest=false`, deliberately).
- Preview endpoints degrade per-document (`available:false` + `reason`; the page GET turns
  the reason into a 404 code), never 500 — host-mode runs without LibreOffice lose only
  PPTX previews (`conversion_unavailable`). `PREVIEW_*` settings are env-only (not in the
  settings drawer); `preview_enabled` is read-only in `GET /api/config`.
- `golden-docs/openapi.json` must be regenerated (`uv run python scripts/export_openapi.py`)
  whenever the API surface changes — a unit test fails otherwise.
- `bun install` silently **converts** a `pnpm-lock.yaml` into `bun.lock` when no `bun.lock`
  exists — never let both lockfiles live in the repo, or the next install re-converts a
  stale file. (`web/` has only `bun.lock` since v3.)

## Package Management

Python: `uv` for everything — `uv sync` (dependency groups: `dev`, `eval`),
`uv run <cmd>`; add dependencies in `pyproject.toml`, then `uv sync`. Torch is pinned to
CPU wheels via `[tool.uv.sources]` (OCR is CPU-only by design; saves ~3 GB).
Frontend (`web/`): `bun` for everything (package manager only — Node stays the runtime) —
never `npm`/`yarn`/`pnpm`.
