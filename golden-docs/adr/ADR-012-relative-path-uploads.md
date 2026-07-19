# ADR-012: Relative-path uploads and the composer's client-side ingest queue

**Status:** Accepted (2026-07-19)

## Context

v2's upload route deliberately **flattened**: `_safe_name` reduced every
client-supplied filename to a bare basename before writing under
`DOCS_PATH` — a correct guard for the corpus page's flat file picker, and
the reason no traversal surface existed. v3's frictionless ingest
(spec_v3 §5) attaches *folders* from the chat composer, and folders make
flattening a correctness bug, not a safety feature: `doc_id` is
`sha256(relative_path + ":" + content_hash)` over the path **relative to
`DOCS_PATH`** — structure is *identity*, not decoration. Flattened,
`q3/notes.md` and `q4/notes.md` collide on one target and the second
silently replaces the first (`replaced=true`). The ingest side has
recursed from v1 (`rglob`); only the upload door was flat.

Relaxing a deliberate guard is the one part of v3 with a real security
surface, so the replacement had to be layered, not clever.

## Decision

- **`_safe_relative_path` replaces `_safe_name` for the upload route
  only**, enforcing spec_v3 §5.2's seven rules in three layers:
    1. structural rejection (rules 1–2): no absolute paths, drive
       letters, or `..` segments; no empty or `.`-prefixed segments (no
       dotfiles, no `.git/`), no Windows reserved device names; control
       characters (NUL included), percent-hex escapes (`%2e`), and any
       character that NFKC-normalizes to a separator or colon are
       rejected outright — nothing is ever decoded or normalized *into
       effect*;
    2. per-segment strictness (rules 3–4): every segment must already
       *be* its own `_safe_name`-sanitized basename — a segment that
       sanitization would change is **rejected, never transformed**, so
       two lookalike paths can never collapse onto one target; whole
       path ≤ 1024 chars, segment ≤ 255, depth ≤ `UPLOAD_MAX_PATH_DEPTH`;
       the final segment passes the unchanged extension vetting (rule 5);
    3. the containment backstop (rule 6), at the write site: after
       joining to `DOCS_PATH`, `resolve()` and assert
       `is_relative_to(docs_root)` — the same check the delete route
       already performs, with `OSError` (dangling symlink) treated as
       outside. It must hold even if rules 1–5 have a hole, and the test
       suite exercises it for real (a symlinked directory inside
       `DOCS_PATH` pointing out of it).
- **`paths` is an optional, positionally-aligned form field** (rule 7:
  `len(paths) == len(files)` or `422 paths_mismatch` — a positional
  contract is checked, not trusted). `paths=None` keeps the flat
  single-file contract **byte-identical** — two doors, two contracts:
  the composer sends structure; `/corpus` keeps its explicit two-step.
- **Batch caps before any byte is written**: `UPLOAD_MAX_FILES` (500) and
  `UPLOAD_MAX_TOTAL_MB` (2048) reject oversized batches with structured
  422s (`too_many_files`, `batch_too_large`); `UPLOAD_MAX_MB` stays the
  per-file cap, and a validator enforces per-file ≤ total. A dragged
  home directory gets a clean 422, not a filled disk.
- **`UploadedFileOut` gains `relative_path`** (nullable — flat uploads
  keep it `null`) rather than overloading `file_name`, plus the per-file
  rejection reasons `invalid_path` and `path_too_deep` beside the
  existing vocabulary.
- **Auto-ingest with a client-side 409 queue**: the composer uploads,
  fires `POST /api/ingest {reingest:false}`, and renders progress as a
  compact chip over the existing `IngestProgressEvent` SSE frames — not
  a modal. When the runner answers `409 ingest_already_running`, the
  composer **holds the attach in a pending queue and re-issues when the
  in-flight run's terminal `status` frame arrives** — the first
  per-status error handling in the frontend.
- **Client-side filtering is summarized, never enumerated**: the folder
  picker's `accept` is unreliable across browsers, so a picked folder
  hands back `.DS_Store`, `.git/`, images — filtered client-side with
  one summary line ("312 files skipped — unsupported type") instead of
  hundreds of rejection rows.

## Rationale

- **Reject-don't-transform** is what makes the per-segment layer
  reviewable: a sanitizer that *fixes* paths has to prove every fix is
  collision-free; a validator that refuses anything sanitization would
  touch has nothing to prove.
- **The backstop is non-negotiable** because rules 1–5 are a vocabulary,
  and vocabularies have holes — `resolve()`-containment is a property of
  the filesystem result itself, mirroring the delete route's precedent.
- **Client-side queueing over a server-side queue**: `IngestRunner` is
  deliberately one-at-a-time, and upload/ingest are decoupled server-side
  — the files are already safely on disk when the 409 arrives, so
  re-issuing costs nothing and loses nothing. A server-side queue would
  move state into the API for a single-user system, and 409 remains the
  honest answer for any other client.
- **The identity claim is asserted, not assumed**: an integration test
  uploads same-named, same-content files to `q3/` and `q4/` over the
  real multipart path and runs the real loader — two documents, two
  distinct `doc_id`s. That is the difference structure makes.

## Consequences

- One click from the composer ingests files *and* folders; nested
  structure survives on disk, so `doc_id`s are collision-free by
  construction and the corpus page's folder grouping reflects what was
  uploaded.
- A composer upload runs `reingest=false`, so it **never clears the
  stale-corpus flag** — only a completed API-driven `reingest=true` run
  does. The corollary is deliberate and unweakened: adding documents
  doesn't re-validate settings drift.
- The wire grew (`paths`, `relative_path`, three new reason/error codes)
  — one `openapi.json` + generated-types regeneration.
- The dotfile rule means a folder attach silently skips `.env`-like
  files and `.git/` internals — the summary line is where that shows.
- Rejected: zip-archive upload (a decompression surface for another
  day), attach-scoped conversations ("ask about *this* doc" without
  ingesting it globally) — both recorded as post-v3 seams.
