# ADR-019: Corpus backups — one zip per backup, restored with no model call

**Status:** Accepted (2026-11-09) — the eleven design decisions were
accepted as recommended on 2026-11-08, and the structure outline added two
more (decisions 12 and 13); this record describes them as built. The
acceptance test is the wiped-stack drill in the
[runbook](../runbook.md#the-wiped-stack-drill).

## Context

`docker compose down -v` deletes every named volume, and with them every
processed byte of both corpora: the chunk rows, their contextual blurbs and
embeddings in pgvector, the BM25 index in Elasticsearch, and the extracted
message graph in `graphdata`. The sources survive (`./docs` and
`./graph-docs` are host binds), so the only way back was to recompute
everything — the LLM over every chunk again, and for the graph days of
extraction (the live graph is 799 thread-days of 64,592 messages). The ask
([#3](https://github.com/luisegarduno/varagity/issues/3)) was a manual
**Export** per corpus into its own folder, and an **Import** that restores
it directly, with no reprocessing, far faster than a rebuild.

## Decision

### 1. Bundles live in a reserved, visible folder inside each corpus root

`docs/varagity-exports/` and `graph-docs/varagity-exports/`
(`varagity.paths.EXPORTS_DIRNAME`). It is where the ticket wants the files,
it stays visible in a file manager, and because it sits inside the host
binds it survives `down -v`. The rest of the system treats it as not-corpus:
ingest discovery skips it (so the CLI's ingest does too), both upload routes
refuse it as a path (`reserved_path`), and on the graph side the build's
scan, the Graph sources listing and the source delete all skip it. Only the
root's own folder is reserved, matched case-insensitively.

Not chosen: a hidden `.varagity-exports/` (hard to find), a reserved suffix
in the corpus root (the graph tab would list bundles as sources), a new
`./backups` bind (a compose change, and not where the ticket wants them).

### 2. "Open the folder" becomes the saved path, Copy path and Download

No browser API can open a folder on the user's machine, and the API runs in
a container that can't reach the desktop. The panel shows where the bundle
landed — checkout-relative (`docs/varagity-exports/<name>`), the form that
reads right in both host and compose mode — with Copy path and a Download
that is a plain attachment link (`FileResponse`,
`Content-Disposition: attachment`), so it needs no CORS.

### 3. The RAG bundle is the rows and their vectors; BM25 is rebuilt

Every `documents` row (plus its `relative_path`), every `chunks` row (every
column but the vector), and the vectors as one matrix, all from **one**
`REPEATABLE READ` snapshot. The BM25 index is deliberately absent: every
field of an Elasticsearch document is a copy of a `chunks` column, so a
restore rebuilds it from the restored rows. Chat history, conversation
folders and settings-drawer overrides are not in any bundle — a bundle backs
up the corpus, not the app.

### 4. A bundle is a zip: JSONL rows, a raw float32 matrix, `bundle.json` last

`documents.jsonl` and `chunks.jsonl` (deflated), `embeddings.f32` (N ×
`EMBEDDING_DIM` little-endian float32, stored — row *i* is line *i*'s
vector, by construction), and `bundle.json`, the manifest, written **last**
so it can record every member's size and sha256. The zip's central
directory lets the list read the manifest without unpacking the rest. A
bundle is written to a hidden `.{name}.partial`, synced and `os.replace`d
into place, so a listing sees a whole bundle or nothing. No new
dependency — not pyarrow, not pickle (loading one runs code), not `pg_dump`
(absent from the api image) or an ES snapshot repository (`path.repo` and a
new mount).

### 5. The graph bundle is the engine's workdir, verbatim

All of LightRAG's namespace files and the repo's two sidecars, under one
folder named for the engine (`lightrag/…`), `*.tmp*` skipped. The LLM
response cache rides along because a later incremental build rebuilds the
entities a changed thread-day shares from it, and the build manifest rides
along so the next build stays a diffing upsert. The export copies under
`GraphService.writing()`, the lock every build takes, so no file changes
mid-copy while chat keeps reading; the graph-stale flag is read last. The
manifest records the engine's file format as the engine reports it
(`GraphEngine.storage_format()`, from package metadata — the backup code
never names the library).

### 6. Restore replaces, and fails safe

The GUI asks first whenever the corpus isn't empty (after `down -v` there is
nothing to replace, so it doesn't).

- **RAG** — one Postgres transaction: `TRUNCATE`, the HNSW index dropped
  (its definition saved from `pg_indexes`), binary `COPY` of both tables
  (each member's sha256 checked again as it streams), the index rebuilt,
  `ANALYZE`, `COMMIT`. The commit is where the restore takes effect; any
  failure before it rolls back to the old corpus. The BM25 index is then
  recreated and refilled with `streaming_bulk`; if that fails the rows are
  restored and the run says so — restoring the same bundle again repairs it.
- **Graph** — extract, swap, open. The files are extracted (sha256
  re-checked, each synced) into `<workdir>.restore-<run>` beside the
  workdir, on the same volume. `GraphService.replace_workdir` then, under
  the write lock throughout and the open lock for the renames (no lazy open
  can land on a half-swapped folder): closes the session — finalizing the
  engine's storages and resetting LightRAG's process-global state — moves
  the live workdir to `<workdir>.previous-<run>`, moves the staged folder
  into its place, and opens it (LightRAG asserts its vectors'
  `embedding_dim`). The **commit point** is renaming `.previous-<run>` to
  `.replaced-<run>`; only then is the replaced graph deleted. Any failure
  before it undoes the swap and reopens the previous graph.
- **During a restore**, chat against that corpus answers `503
  corpus_restoring` — a graph turn before its session is opened — and
  deletes in that corpus answer `409 corpus_busy` (a graph source delete
  would set the stale flag the restore is about to rewrite).

Not chosen: empty-only restores (rolling back would mean deleting
everything first, and the graph has no "delete all") and merging (old and
new versions of a changed file side by side; graph files can't be merged
without re-extraction).

### 7. Strict compatibility: refused, never converted

A restore is refused with `422 bundle_incompatible`, naming the reason,
unless the bundle's format version is this stack's, its `corpus` is the
target, its embedding model and dimension equal `EMBEDDING_MODEL` and
`EMBEDDING_DIM`, and — for a graph bundle — its engine is the one
`GRAPH_ENGINE` selects, from the same package, at the same **major.minor**
version (a patch release keeps the file format; a `lightrag-hku 1.4.9`
bundle is refused on 1.5.4). Neither store checks the embedding model —
pgvector compares only the vector size, LightRAG only `embedding_dim` — so a
looser rule would restore cleanly and then retrieve the wrong passages. An
unsound bundle (no manifest, an unsafe member — `..`, absolute, backslash,
symlink — an unexpected, missing or unrecorded member, a size that disagrees
with the zip, an empty graph bundle) or a damaged one (a sha256 mismatch) is
`422 bundle_invalid`. Everything that can refuse a bundle runs before the
`202`, so a bad bundle never becomes a failed run.

### 8. A restore re-roots paths and derives the stale flags

Each document's `source` becomes `(DOCS_PATH / relative_path).resolve()`,
exactly as ingest builds it, and every chunk's `metadata.source` follows;
a document from outside the corpus keeps its absolute path. `doc_id`
already hashes the relative path, so identity is portable and the document
list, previews and "also remove the file" keep working across host and
container mode. `_corpus_stale` is set when the bundle was stale at export
or its `ingest_settings` differ from this stack's reingest-affecting
settings (the list reports that `drift` up front); `_graph_stale` takes the
graph bundle's own flag. Either flag is written only once the restore
completed.

### 9. A per-corpus backup runner and a per-corpus write gate

`BackupRunner` is the ingest runner's shape — a `202`, a daemon thread
running a tracked Prefect flow (`rag-export`, `rag-restore`,
`graph-export`, `graph-restore`), and a status stream that replays the
backlog to late joiners — one per corpus. `CorpusGate` is one in-process
claim per corpus, taken by `IngestRunner`, `GraphBuildRunner` and
`BackupRunner` alike, so only one operation runs per corpus; a loser gets
`409 corpus_busy` naming the holder (a second ingest or build keeps its own
`409`). Graph backups also take `GraphService`'s write lock, and the
`reingest=true` build's wipe now does too, closing a race the design found.
No backup operation needs a model service: the preflights are postgres for
a RAG export and both graph operations, postgres and Elasticsearch for a
RAG restore. The composer's 📎 auto-ingest, refused with `corpus_busy`,
waits for the backup run once and then ingests.

### 10. A bundle is picked from the folder's list

`GET /api/backups/{corpus}` lists every zip in the reserved folder, newest
first, with its manifest and verdict; Restore works by name. A bundle
copied in by hand (file manager, `scp`) shows up too; an in-flight
`.partial` never does. No browser upload — it would only matter when the
browser runs on another machine, and would need its own size cap.

### 11. API and GUI only — no CLI command

The graph can't have one (the API process is its single writer), and one
shape for both corpora keeps the runbook simple. The runbook carries `curl`
recipes instead.

### 12. The list's verdict is cheap; a restore always hashes

The list reads only each zip's central directory and `bundle.json` —
version, corpus, member rules, each member's recorded size against the
zip's, the embedding space, the engine format — and never hashes: a graph
bundle is about 150 MB, panels reload, and bundles pile up. A restore
hashes every member before it touches a store, so a damaged bundle can
never restore. A sha256 failure is remembered in-process by corpus, name,
size and mtime, so the list then shows that bundle as damaged instead of
offering it again. What remains is a bundle damaged inside an intact zip
with unchanged sizes, which lists as restorable until its first Restore
refuses it.

### 13. The RAG corpus has a cross-process advisory lock

The gate is in-process, but RAG writers also run in their own processes —
`uv run main.py ingest`, and `main.py chat`, which ingests on startup. Every
ingest holds a Postgres advisory lock on one fixed key **shared** for its
whole run (so ingests still overlap each other); a restore holds it
**exclusive** from before the `TRUNCATE` until the BM25 rebuild ends. A CLI
ingest refused by a restore exits with one line and exit 1; a restore
refused by a CLI ingest answers `409 corpus_busy` before the `202`. The lock
dies with its database session. Without it, a CLI ingest could write rows
the BM25 rebuild misses, or index into the gap between the index's delete
and re-create — where Elasticsearch auto-creates the index with default
mappings that `create_index()` then keeps for good. The graph needs no such
half: its only writer is the API process.

### As built: an interrupted graph swap is healed when the service is constructed

A crash can leave the swap's folders beside the workdir. Constructing
`GraphService` — once per process, before anything can read, open, wipe or
restore the workdir — finishes the job: a `.previous-<run>` means the swap
never committed, so the previous graph moves back (an unconfirmed restored
workdir is removed first); `.replaced-<run>` and `.restore-<run>` folders
are debris and are removed. The heal sits in construction rather than in
`session()` (where the structure outline first placed it) for three
reasons: the status route and the export check `workdir.is_dir()` without
opening a session, so a heal that waited for an open would leave the graph
looking unbuilt; a `reingest=true` build wipes the workdir before its first
open, so a heal in `session()` would mistake that wipe for a crash and
resurrect the pre-restore graph; and a lazy open (a status poll) during a
restore's extraction would delete the staging folder being extracted.

### As built: a restore hashes once, in its run (supersedes part of §7 and §12)

§7 promised that everything able to refuse a bundle runs before the `202`,
and §12 that a restore hashes every member before it touches a store. As
built first, the restore route did that — and the run then read the whole
bundle again to load it, hashing every member a second time on the way. A
review counted the cost: every restore read the bundle twice, the `POST`
blocked for the whole first pass with no progress (a multi-GB RAG bundle
held it for minutes, long enough for a proxy to time it out), and the corpus
gate wasn't held meanwhile, so an ingest could start and turn minutes of
hashing into a `409`.

The second pass was the one that mattered. The run's verifying streams
already guarantee §12's real goal — a damaged bundle can never restore —
because nothing takes effect until every member checked out: the RAG load
is one transaction that commits only after the last member's digest
matched, and the graph's files are extracted into a staging folder that is
swapped in only after the last one did. So the route now runs the cheap
verdict alone (the list's own check), plus a damage an earlier restore of
the same file found, and the run's single read does all the hashing, with
progress. The price is §7's other promise: a bundle damaged *inside* an
intact zip — the one defect the cheap verdict can't see — now fails its
first **run** (the corpus untouched) instead of answering `422`. The runner
remembers that failure by name, size and mtime, exactly as the route did,
so the list shows the bundle as damaged and the route refuses it with `422
bundle_invalid` from then on.

### As built: the stale flag follows the rows (refines §8)

§8 wrote either flag "only once the restore completed". A RAG restore whose
BM25 rebuild fails after its commit did not complete — but its rows are the
bundle's by then, so leaving the previous corpus's flag would misdescribe
them until the next restore. The `rag-restore` flow therefore also writes
the derived flag when only the rebuild failed (`BM25RebuildFailed` carries
the restored bundle's manifest), and still writes nothing after a restore
that rolled back.

### As built: reads of the tables answer 503 during a RAG restore

The restore's `TRUNCATE` holds `documents` and `chunks` until it commits —
through the whole binary `COPY` and the HNSW build. Only chat and the
deletes refused during that window at first; the document list, the page
previews and a settings change's emptiness count queued behind the lock for
the length of the load, each parking a worker thread. Every route that
queries the tables now answers `503 corpus_restoring` while a restore holds
the corpus (`require_rag_readable`), and a settings change made during a
restore flags the corpus stale without counting it: the restore ends by
writing the flag its bundle implies against the settings current then, and
a restore that rolls back leaves rows the change does make stale.

## Consequences

- **A wiped stack comes back in minutes.** A RAG restore costs one `COPY`
  and two index builds; a graph restore costs an unzip and a rename — the
  multi-day extraction is never repeated.
- **`graphdata` needs room for two graphs during a swap** (the staged copy
  beside the live one), and `docs/` / `graph-docs/` grow by every bundle
  kept. Bundles are ordinary files: there is no GUI delete.
- **Graph bundles are tied to the engine's major.minor.** Upgrading
  `lightrag-hku` across a minor version makes older graph bundles
  unrestorable (listed, with the reason): export again after an upgrade.
- **Chat history is not protected.** `down -v` still wipes conversations,
  folders and settings overrides; backing those up would be a different
  feature.
- **The gate is per process.** For RAG the advisory lock extends it to CLI
  ingests; nothing else may write `graphdata` (the single-writer rule
  stands).
- **A graph turn already streaming when a restore swaps can end in an
  error event**; new graph turns are refused for the whole restore, and the
  swap itself takes seconds.
- **Restores are mode-faithful.** A host-mode restore re-roots sources onto
  `./docs`, which is right for host mode — and the compose API would then
  list every document flat until it restores the bundle itself.
