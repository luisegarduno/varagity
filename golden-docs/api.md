# HTTP API

The FastAPI service (`varagity/api/`, port `8000`) is the system's single
backend surface: the web GUI talks only to it, and the CLI is its peer over
the same Prefect flows (spec_v2 §4). The layer is **async at the edge, sync
flows underneath** — routes run the pipeline in a worker threadpool rather
than rewriting it (spec_v2 §4.1). The wire contract lives in
`varagity/api/schemas.py` as pydantic models; everything below the
hand-written sections is **auto-rendered from `openapi.json`**, a checked-in
snapshot of the app's schema (see [Keeping this page
honest](#keeping-this-page-honest)).

Two things OpenAPI cannot express are documented by hand here: the SSE
event protocols and the error-envelope conventions.

## The error envelope

Every non-2xx response carries the same structured body (spec_v2 §4.1) —
the GUI maps `code` to an actionable banner, and the shape holds for
*every* failure, including unhandled exceptions (a pure-ASGI middleware
seated inside CORS envelopes those as `internal_error`, so cross-origin
browsers see the real error instead of `TypeError: Failed to fetch`):

```json
{"error": {"code": "es_unreachable", "message": "elasticsearch unreachable — is the stack up? (…)"}}
```

The code vocabulary, as built:

| Code | Status | Raised by |
|---|---|---|
| `validation_error` | 422 | Any request-body/parameter validation failure (field errors ride in `message`). |
| `unknown_retrieval_method` | 422 | `POST /api/chat` override naming no registered retriever. |
| `unknown_chat_engine` | 422 | `POST /api/chat` override naming no registered chat engine. |
| `unknown_setting` / `invalid_settings` | 422 | `PATCH /api/settings` — unknown name / value failing the `Settings` validators (linked settings validate as a merged whole). |
| `no_file_stored` | 422 | `POST /api/documents` when every file in the batch was rejected. |
| `paths_mismatch` | 422 | `POST /api/documents` when the `paths` form field doesn't pair 1:1 with `files` (the folder-upload positional contract is checked, not trusted). |
| `modified_mismatch` | 422 | `POST /api/documents` when the `modified` form field (per-file last-modified epoch-ms; restamps each stored file's mtime so `file_modified_at` provenance survives the upload) doesn't pair 1:1 with `files`. Malformed *entries* are ignored per file, not errors. |
| `too_many_files` / `batch_too_large` | 422 | `POST /api/documents` when the batch busts `UPLOAD_MAX_FILES` / `UPLOAD_MAX_TOTAL_MB` — rejected before any byte is written. Per-file problems (`invalid_path`, `path_too_deep`, `extension_not_allowed`, `file_too_large`, `reserved_path` — a path under, or a flat file named, the reserved `varagity-exports` backup folder — …) are **not** errors: they ride `UploadedFileOut.reason` inside the 201. |
| `nothing_to_export` | 422 | `POST /api/backups/{corpus}/export` when the corpus holds no documents — an empty bundle is not a backup. |
| `bundle_invalid` | 422 | `POST /api/backups/{corpus}/restore` for a bundle that isn't sound: no readable `bundle.json`, an unsafe member (`..`, an absolute path, a symlink), an unexpected, missing or unrecorded member, a member whose recorded size differs from the zip's, a vector matrix that disagrees with the chunk count — or a member whose bytes don't match their recorded sha256 (*damaged*: the list then shows that bundle as damaged until the file changes). Nothing was touched. |
| `bundle_incompatible` | 422 | `POST /api/backups/{corpus}/restore` for a sound bundle this stack must not restore: another bundle format version, another corpus, or vectors from another embedding model or dimension. Refused, never converted — neither store checks the model, so a mismatch would restore cleanly and then retrieve the wrong passages. |
| `bundle_not_found` | 404 | `GET /api/backups/{corpus}/files/{name}` and `POST /api/backups/{corpus}/restore` for any name that is not a bundle file inside the corpus's `varagity-exports/` folder — hidden and in-flight `.partial` files, non-`.zip` names, and names (or symlinks) leading out of the folder included. |
| `conversation_not_found` | 404 | `POST /api/chat` (pre-stream) and the conversation routes. |
| `document_not_found` | 404 | `DELETE /api/documents/{doc_id}` and the two `…/preview/*` routes — the bulk `POST /api/documents/delete` reports unknown ids in `not_found` rather than failing the batch. |
| `preview_disabled`, `unsupported_type`, `file_missing`, `file_changed`, `conversion_unavailable`, `conversion_failed`, `page_out_of_range` | 404 | `GET /api/documents/{doc_id}/preview/page/{page}` only — an `<img>` can't read a JSON envelope, so the page route turns each degrade reason into a 404 code. The locate route reports the same conditions inside a **200** envelope instead ([below](#the-preview-pair-degradable-by-design)). |
| `ingest_already_running` | 409 | `POST /api/ingest` while a run is in flight (one run at a time). |
| `corpus_busy` | 409 | One writer per corpus. `POST /api/backups/{corpus}/export` and `…/restore` while another operation holds the corpus's write gate — a second backup run included; `POST /api/ingest` while a backup export or restore holds the RAG gate; the two document delete routes while a backup export or restore holds it. A restore is also refused while an ingest *in another process* (a CLI `main.py ingest` / `chat`) holds the corpus's advisory lock. The message names the holder (`a restore is running`). |
| `corpus_restoring` | 503 | `POST /api/chat`, before the stream opens, while a backup restore is replacing the corpus that would answer the turn (a graph turn the graph answers is unaffected by a document restore). Ask again once the restore finishes. |
| `<service>_unreachable` | 503 | Dependency preflights and per-request store construction: `postgres_unreachable`, `es_unreachable`, `llamacpp_unreachable`, `infinity_unreachable`. |
| `docs_path_not_writable` | 500 | `POST /api/documents` when `DOCS_PATH` rejects writes (the container-UID gotcha — see the runbook); `POST /api/backups/rag/export` when `docs/varagity-exports/` can't be created or written (the same gotcha, or a *file* squatting on the folder's name). |
| `internal_error` | 500 | Any unhandled exception (enveloped by the middleware). |
| `pipeline_error` | — | **In-band SSE only**: the chat stream's `error` event when the pipeline fails after the 200 flushed. |
| `bad_request`, `not_found`, `method_not_allowed`, `service_unavailable`, `http_<status>` | 400/404/405/503/other | Status-derived fallbacks for framework-raised errors (unknown path, wrong method, …). |

## The chat stream — `POST /api/chat`

One question in, one `text/event-stream` out (spec_v2 §4.3). The transport
is a **POST** — the browser `EventSource` API is GET-only, so clients
consume it with `fetch` plus an SSE parser (the web app uses
`eventsource-parser`); each frame is standard SSE framing,
`event: <name>` + `data: <json>`.

Event order:

```
retrieval → (reasoning | token | stats)* → done | error
```

The `retrieval` frame fires once retrieval (+rerank) completes, **before
any answer token** — evidence before prose (spec_v2 §4.3). That ordering is
the transparency story: the browser's provenance panel is populated while
the answer is still streaming, and it is the flow's natural shape
(`on_retrieved` fires before generation starts). `reasoning` and `token`
deltas then interleave in stream order — a reasoning model's
`<think>…</think>` content is classified fragment-by-fragment
(`ThinkStreamSplitter`) into `reasoning` events, everything after into
`token` events. Throttled `stats` frames (live decode throughput) may
interleave with the deltas — **only when the model server reports its own
timings**. llama.cpp does; a backend that doesn't simply produces a stream
with zero `stats` frames, so clients must treat their absence as normal.

| Event | Payload | Fields |
|---|---|---|
| `retrieval` | `RetrievalEvent` | `chunks` (list of `RetrievedChunk`, best first, each with metadata and — when the method fills it — a `RetrievalTrace`), `method`, `top_k`, `reranked_to` (`RERANK_TOP_N` when the `reranked` method narrowed the list, else `null`), `condensed_query` (v3 — the standalone search query the chat engine actually retrieved with, `null` whenever the turn wasn't condensed: a first turn, the `simple` engine, the kill switch, or the raw-query fallback; [ADR-011](adr/ADR-011-chat-engine-condense.md)). |
| `reasoning` | `DeltaEvent` | `delta` — the next `<think>` fragment, stream order. |
| `token` | `DeltaEvent` | `delta` — the next answer fragment. |
| `stats` | `StatsEvent` | `tokens_per_second` (decode throughput so far, cumulative average — it settles rather than jitters), `completion_tokens` (tokens decoded so far). Warmup-gated (no frame before 8 decoded tokens — the model server's first readings compute absurd rates) and throttled to ≥250 ms apart. |
| `done` | `DoneEvent` | `message_id`, `conversation_id` (the one just created when the request named none), `answer` (full, `<think>`-stripped — authoritative; streamed deltas are best-effort display), `usage` (`prompt_tokens`, `completion_tokens`, `latency_ms` keyed `retrieval`/`generation`/`total`, `tokens_per_second` — the model server's own final decode rate, `null` when it reports no timings). |
| `error` | `ErrorEvent` | `code` (`pipeline_error`), `message`. |

**Failure surfaces split by stream state.** Everything detectable before
streaming is a clean structured status, checked cheapest-first: body shape
(422 `validation_error`) → retrieval-method / chat-engine resolution (422
`unknown_retrieval_method` / `unknown_chat_engine`) → the corpus that would
answer being mid-restore (503 `corpus_restoring`) → dependency preflight (503
`<service>_unreachable`, probing `postgres`, `elasticsearch`, `llamacpp`,
`infinity` — prefect is deliberately absent: without a server, flows fall
back to an ephemeral in-process API, so chat works untracked) → conversation
existence (404 `conversation_not_found`). Once the 200 flushed, HTTP status
can't change — anything the pipeline raises mid-stream arrives as the
in-band `error` event instead.

**Disconnect semantics.** A client disconnect cancels the response
generator, which flips an abort flag the flow polls **between tokens**; the
LLM stream closes and the GPU is freed (spec_v2 §4.3 cancellation). Aborted
turns persist *nothing* — the turn (both messages, evidence snapshots,
timings) is persisted at `done`, whose ids prove it. Conversation
auto-titling runs fire-and-forget after `done` and never delays it.

## The ingest status stream — `GET /api/ingest/status`

`POST /api/ingest` returns `202` with a run handle (`IngestRunOut`)
immediately; the flow — the same tracked Prefect ingest flow the CLI runs —
executes on a background thread (spec_v2 §4.2). **One run at a time**: a
second `POST` while one is in flight is a `409 ingest_already_running`. The
preflight 503-checks `postgres`, `elasticsearch`, `infinity`, plus
`llamacpp` only when `CONTEXTUALIZE` is on.

`GET /api/ingest/status` streams the run's feed with **replay-from-frame-one
semantics**: a subscriber always gets the full backlog first, then live
events while the run is still going — connecting mid-run or after
completion renders the same picture the CLI's `rich` display showed live.
With no run ever started in this API process, the stream is a single idle
`status` frame (`run: null`) and closes.

| Event | Payload | Fields |
|---|---|---|
| `status` | `IngestStatusEvent` | `run` (`IngestRunOut`: `run_id`, `state` `running`/`completed`/`failed`, `reingest`, timestamps, terminal `summary` counters, flow-level `error`) — always the first frame (snapshot) and the last (terminal). |
| `progress` | `IngestProgressEvent` | `stage` (`discover` \| `parse` \| `chunk` \| `contextualize` \| `embed` \| `store` \| `file_done`), `file`, `outcome` (`file_done` only: `ingested`/`skipped`/`no_text`/`failed`), `current`, `total`, `files_done`, `files_total`. |
| `log` | `IngestLogEvent` | `level`, `message` — relayed `varagity.ingest` records (skips, no-text warnings, failure heads), pinned to `INFO` for the run. |

One `progress` frame per stage transition — except `contextualize`, the
ingest's long pole (one LLM blurb per chunk), which additionally ticks
per-chunk via `current`/`total`, so the browser's progress bar moves at the
same granularity as the terminal's.

## The backup status stream — `GET /api/backups/{corpus}/status`

`POST /api/backups/{corpus}/export` writes one backup bundle — a zip of the
corpus's processed rows, see [Data model](data-model.md#backup-bundles) —
into the corpus folder's reserved `varagity-exports/`. It returns `202` with
a run handle (`BackupRunOut`) at once, and the tracked `rag-export` flow
runs on a background thread, the same shape as ingest. Everything that can
refuse an export is checked **before** the 202, so a refusal is a clean
status, never a failed run:

1. The preflight: `503 postgres_unreachable` (an export reads the rows and
   the stale flag, and calls no model service).
2. `422 nothing_to_export` when nothing is ingested.
3. `500 docs_path_not_writable` when the exports folder can't be written.
4. `409 corpus_busy` while any operation holds the corpus.

The last one is a per-corpus **write gate**, not a per-runner flag: one
operation per corpus at a time — ingest, export and restore all claim it, so
a second backup run is refused too. Only `rag` is a valid `{corpus}` today;
anything else is a `422 validation_error`.

`POST /api/backups/{corpus}/restore` with `{"name": "<bundle>"}` starts a
restore the same way. A restore **replaces** the corpus: its rows in both
stores become the bundle's, re-rooted onto this stack's `DOCS_PATH`, with no
model call, and the BM25 index is rebuilt from them. Its refusals, in order,
again all before the 202:

1. `503 postgres_unreachable` / `es_unreachable` — the two stores it writes.
2. `404 bundle_not_found` — no such bundle in the folder.
3. `409 corpus_busy` — another operation holds the corpus. Checked before the
   expensive step below.
4. `422 bundle_invalid` / `bundle_incompatible` — the bundle's verdict, plus
   the sha256 of **every member** (the list never hashes; only this does). A
   sha256 mismatch is remembered in-process by name, size and mtime, so the
   list then reports that bundle as damaged.
5. `409 corpus_busy` — an ingest in another process holds the corpus's
   advisory lock.

While a restore runs, `POST /api/chat` against that corpus answers `503
corpus_restoring`, `GET /api/documents` waits for the restore's commit, and
the document deletes answer `409 corpus_busy`.

`GET /api/backups/{corpus}/status` has the ingest stream's **replay-from-frame-one**
semantics. A subscriber gets the run's whole backlog, then live events until
the terminal frame. With no backup run in this API process it is a single
idle `status` frame (`run: null`).

| Event | Payload | Fields |
|---|---|---|
| `status` | `BackupStatusEvent` | `run` (`BackupRunOut`: `run_id`, `corpus`, `operation` `export`/`restore`, `state` `running`/`completed`/`failed`, timestamps, run-level `error`). A completed export adds `bundle` (the file name), `saved_to` (`docs/varagity-exports/<name>`, the checkout-relative path the panel's Copy path copies) and `summary` (`counts`, `size_bytes`, `wall_clock_s`). A restore names its `bundle` from the first frame; its `summary` counts what it wrote (`documents`, `chunks`, `bm25`) and adds `stale`, the corpus-stale flag it left. The first frame is the snapshot and the last is the terminal state. |
| `progress` | `BackupProgressEvent` | `stage` — an export's `snapshot` \| `documents` \| `chunks` \| `finalize`, a restore's `validate` \| `load` \| `index` \| `bm25` — plus `current`/`total` rows for the counted stages (`documents`, `chunks`; `load` and `bm25` count chunks). Throttled to about 50 frames per stage, so a large corpus doesn't flood the stream. |
| `log` | `BackupLogEvent` | `level`, `message`: relayed `varagity.backup` records. These include what was exported or restored, a swept partial from an interrupted export, and a warning when rows record an embedding model other than `EMBEDDING_MODEL`. |

The terminal transition is atomic: the run turns terminal, its last frame
joins the backlog, and the corpus is released together. A client that
starts the next export as soon as it sees `completed` never races a
spurious `corpus_busy`.

The other two backup routes are plain:

- `GET /api/backups/{corpus}` lists the folder's bundles, newest first. Each
  entry carries its `bundle.json` manifest, read alone through the zip's
  central directory, or `manifest: null` for a zip that isn't a readable
  Varagity backup — and the **cheap restore verdict**: `restorable`, the
  `reason` when it isn't (the same reasons a restore's 422 gives, worded for
  the panel), and `drift`, the reingest-affecting settings the bundle was
  made under that differ from this stack's (restoring it flags the corpus
  stale; drift never refuses). The verdict reads only the zip directory and
  the manifest, so a bundle damaged inside an intact zip lists as
  restorable until its first restore refuses it.
- `GET /api/backups/{corpus}/files/{name}` returns one bundle as
  `application/zip` with `Content-Disposition: attachment`. The panel's
  Download is a plain link, so no cross-origin fetch (and no CORS) is
  involved.

## The preview pair — degradable by design

The evidence panel's page previews
([ADR-010](adr/ADR-010-document-page-preview.md)) ride two routes with an
asymmetric error convention. `POST /api/documents/{doc_id}/preview/locate`
answers **200 for every degradable condition** — `available:false` plus a
machine `reason` (`preview_disabled` | `unsupported_type` | `file_missing` |
`file_changed` | `conversion_unavailable` | `conversion_failed` |
`no_match`) — reserving 404 for an unknown `doc_id`; the GUI maps any
`available:false` to its full-text fallback, so a missing LibreOffice or an
edited-on-disk file is a per-document degrade, never a failed request.
`GET /api/documents/{doc_id}/preview/page/{page}` serves the rendered PNG
with `Cache-Control: public, max-age=31536000, immutable` — sound because
`doc_id` is content-hashed **and** the route re-verifies the on-disk
`content_hash` before rendering (a drifted file 404s `file_changed` rather
than serving a lying image). Clients only request pages a successful locate
named, so this route's 404s are the honest edge signal, carrying the degrade
reason as the error code.

## Keeping this page honest

The route reference below renders `openapi.json`, a snapshot regenerated by
`uv run python scripts/export_openapi.py` (it pins `METRICS_ENABLED=true`,
because that setting gates `/metrics`' presence in the schema) and
**drift-guarded** by `tests/unit/test_openapi_snapshot.py` — the unit suite
fails whenever the app's live schema stops matching the checked-in copy.
The frontend's `web/lib/types.ts` is generated from the same schema
(`bun run gen:types`), and the app factory merges the SSE payload models into
`components/schemas` even though no route returns them directly — the
schemas named in the tables above are all in the reference below. A running
stack also serves the live interactive docs at `http://localhost:8000/docs`.

## Route reference (auto-rendered)

[OAD(./openapi.json)]
