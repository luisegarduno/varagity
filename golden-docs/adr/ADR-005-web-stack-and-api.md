# ADR-005: The v2 web GUI + HTTP API stack

**Status:** Accepted (2026-07-14)

One record for the cluster of stack decisions behind the v2 centerpiece —
the browser chat surface and the HTTP API underneath it (spec_v2 §4, §14).
Like [ADR-003](ADR-003-vertical-build-and-ops-choices.md), each call was
small alone; together they define how the web surface works and scales.

## 1. FastAPI at the edge, the sync flows underneath

**Context.** The pipeline is deliberately synchronous (openai SDK, psycopg,
elasticsearch, Prefect flows run in-process). The GUI needs streaming, which
tempts an async rewrite.

**Decision.** A **FastAPI** app inside the package (`varagity/api/`), async
only at the edge: routes run the *unchanged* sync flows in a threadpool
(`run_in_threadpool`), and an event bridge marshals stream frames back onto
the loop. Explicitly **not** an async pipeline rewrite (spec_v2 §14 #4).

**Consequences.** The CLI and the API stay peer front-ends over the same
flows — the API can never diverge from what the pipeline actually does. The
event loop stays free to stream while a flow occupies one worker thread;
client disconnect aborts generation between tokens (nothing persisted on an
aborted turn). The cost: pipeline calls block threads, so concurrency is
bounded by the threadpool — acceptable for the single-user posture (§4).

## 2. The 2026 frontend stack, with generated types

**Context.** spec_v2 §4.5 committed to Next.js + TypeScript + Tailwind +
shadcn/ui, but its phrasing ("Radix primitives", `tailwind.config.ts`)
predated the current toolchain (plan decision #2 treats it as dated, not
prescriptive).

**Decision.** **Next.js (App Router) + TypeScript + Tailwind v4 + shadcn/ui
on Base UI.** Tailwind v4 is CSS-first: design tokens live in an `@theme`
block in `web/app/globals.css`; there is no `tailwind.config.ts`. Frontend
types are **generated** from the API's OpenAPI schema (`pnpm gen:types` via
`openapi-typescript` → `web/lib/types.ts`, never hand-edited); the API even
merges its SSE payload models into the schema (they're not route returns) so
stream frames are typed too.

**Consequences.** The wire contract can't drift — a schema change is a
regenerate, not a hunt. The API is frontend-agnostic, so a Vite + React SPA
remains the drop-in alternative (spec_v2 §14 #1); Radix stays one
`shadcn init -b radix` away. `web/` runs its own toolchain (pnpm, Vitest,
Playwright), disjoint from `uv`.

## 3. SSE over POST; evidence before the prose

**Context.** Chat streaming is one-directional per turn. WebSockets buy
bidirectionality nobody needs and lose easy proxying; the native browser
`EventSource` is GET-only and can't carry the JSON question body.

**Decision.** `POST /api/chat` returns `text/event-stream` (native
`EventSourceResponse`; the `fastapi>=0.135` pin exists for it), consumed in
the browser with `fetch()` + **`eventsource-parser`**. The event protocol is
`retrieval → reasoning/token deltas → done` (or `error`): the provenance
payload lands **before** any answer token — the transparency story as wire
order. Errors after the 200 has flushed are **in-band** `error` events (the
status line is gone; mid-stream failures must ride the stream).

**Consequences.** The evidence panel renders while the answer streams; the
CLI, web panel, and `message_sources` snapshots all consume the same
`RetrievedChunk` + trace data ([ADR-006](ADR-006-reranking-wired.md)).
Pre-stream failures still surface as clean structured `503`s from a
dependency check that runs before the stream opens.

## 4. Single-user, local, no auth — with persistence that survives reingest

**Context.** Owner-confirmed scope (spec_v2 §14 #2): one user, LAN, no
login. But history matters, and reingest rewrites `chunk_id`s.

**Decision.** No user/role/session tables; conversations persist in the
existing Postgres (`conversations`/`messages`/`message_sources`).
`message_sources` **snapshots** each answer's evidence (content, context,
source, trace JSONB) and keeps `chunk_id` as a deliberately **soft**
reference — no FK to `chunks`.

**Consequences.** Historical conversations still explain themselves after a
reingest deletes or renumbers their chunks — the snapshot is what produced
*that* answer, by design. Multi-user is post-v2 (spec_v2 §15): the seam is
`conversations.owner` + corpus-scoped queries, nothing pre-built. The
dev-only security posture (open ES/pg/llama.cpp on the LAN) is unchanged and
documented in the [runbook](../runbook.md#security-posture-dev-only).

## 5. Provenance panel + inline citations, not in-browser document preview

**Context.** The transparency ceiling (kotaemon-style) is click-a-citation →
the original document with the span highlighted. That needs ingest-time
char-offset + per-chunk page provenance v1 never tracked.

**Decision.** Owner-confirmed (spec_v2 §14 #3): ship the **evidence panel**
(per-chunk score, `sem/bm25/fused/rerank` badges, context blurb, expandable
text with client-side term highlighting) plus **inline citation chips**, and
defer document preview. v2 deliberately does not add ingest offsets; the
seam stays open (chunker `start_index` + loader page map, then a PDF.js/text
viewer — spec_v2 §15).

**Consequences.** Ingest stayed untouched by the GUI build. Citations that
match no retrieved source are flagged ("not in evidence") — a cheap
grounding-drift signal. One rendering landmine is load-bearing: line-initial
`[SOURCE]: /path` is a CommonMark link-reference *definition* and vanishes
when rendered, so `web/lib/citations.ts` rewrites citations to chips
**before** markdown parsing.

## 6. A hand-rolled migration runner over Alembic

**Context.** `schema.sql` runs only on first boot (`docker-entrypoint-initdb.d`);
existing `pgdata` volumes needed the v2 tables. The codebase has no ORM, and
the v2 delta is two migration files of additive DDL
(`001_conversations.sql`, `002_app_settings.sql`).

**Decision.** Ordered, idempotent SQL in `varagity/stores/migrations/NNN_*.sql`,
tracked in a `schema_migrations` table, applied by the API on startup — one
transaction per file, so a failure rolls back atomically and a re-run
retries just that file. Alembic is the recorded heavier alternative (plan
decision #8): autogeneration has no models to diff against, and its
env/versioning machinery outweighs this schema's needs.

**Consequences.** `schema.sql` stays the fresh-install fast path and must be
kept in sync with the migrations by hand — the accepted cost. Migrations run
from the API lifespan only (the CLI doesn't migrate); an unreachable
Postgres at boot logs and continues, a *failing* migration fails startup
loudly. Revisit Alembic if migrations grow branches or destructive changes.

## 7. Smaller calls recorded with their rationale

- **Single uvicorn worker** (plan decision #11): one process = one
  Prometheus registry — no `PROMETHEUS_MULTIPROC_DIR` machinery
  ([ADR-007](ADR-007-observability-stack.md)). Scale by container replicas,
  not workers.
- **`NEXT_PUBLIC_API_URL` is a build-time constant** — Next.js inlines it,
  and it rides a compose build arg, so repointing the API means
  `docker compose build web`. LAN browsers need the host's address there
  *and* in `API_CORS_ORIGINS`.
- **CORS pinned to the web origin** (`API_CORS_ORIGINS`, default
  `http://localhost:3000`) — the browser is the only cross-origin caller. A
  pure-ASGI `StructuredServerErrors` middleware sits inside CORS so even
  unhandled 500s stay browser-readable `{error: {code, message}}` envelopes.
- **Auto-titling is fire-and-forget** — the first-question title LLM call
  runs after `done` is queued and can never block the stream.
