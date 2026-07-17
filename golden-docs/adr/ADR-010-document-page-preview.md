# ADR-010: Document page preview via on-demand server-side locate + render

**Status:** Accepted (2026-07-16)

## Context

The transparency ceiling [ADR-005 §5](ADR-005-web-stack-and-api.md) deliberately
deferred: click a source and see *the page of the original document that
produced it*, with the chunk highlighted — Kotaemon's signature affordance.
The deferral note assumed the feature needed ingest-time provenance (per-chunk
page/char offsets v1 and v2 never captured; `ChunkRecord.page` is
document-level). The owner-confirmed decisions (2026-07-16): server-rendered
pages with a client overlay (not client-side PDF.js), LibreOffice inside the
api image for PPTX (not a sidecar, not deferred), inline previews in the
evidence rail with a click-to-enlarge dialog.

## Decision

- **Locate at preview time, not ingest time** (`varagity/preview/`): when an
  eligible chunk expands, `POST /api/documents/{doc_id}/preview/locate` scores
  every page of the source by **word-trigram containment** of the chunk's
  de-decorated markdown (unigrams for chunks under 8 words), then computes
  highlight rectangles by pdfium text search over short sentence/window
  snippets — merged per line, deduplicated, capped at 300, normalized to
  `[0, 1]` with a top-left origin so the client does percentage math only.
  Best coverage below `PREVIEW_MIN_COVERAGE` reports `no_match`.
  `GET /api/documents/{doc_id}/preview/page/{page}` renders the page PNG at
  `PREVIEW_RENDER_WIDTH`, marked `Cache-Control: public, max-age=31536000,
  immutable` — sound because `doc_id` is content-hashed and both routes
  re-verify the on-disk `content_hash` first (a drifted file degrades honestly
  instead of previewing the wrong bytes).
- **Eligibility**: digital PDFs (`extraction != "ocr_fallback"`) and `.pptx`.
  PPTX decks convert once per container lifetime via headless **LibreOffice
  Impress in the api image** (`soffice --headless --convert-to pdf`, throwaway
  `-env:UserInstallation` profile, module lock, `PREVIEW_CONVERT_TIMEOUT_S`),
  cached under `doc_id` in the container's temp dir; Impress maps slide N to
  PDF page N — the same identity Docling relies on ([ADR-009](ADR-009-modality-expansion.md)) —
  so locate/render work unchanged on the converted artifact. The image adds
  `libreoffice-impress` plus metric-compatible fonts (Liberation, Carlito,
  Caladea, DejaVu — Gotenberg's slim set): **+0.31 GB** measured.
- **Every failure degrades, nothing 500s**: the locate answers `200` with
  `available:false` + a machine `reason` (`preview_disabled` |
  `unsupported_type` | `file_missing` | `file_changed` |
  `conversion_unavailable` | `conversion_failed` | `no_match`); the image
  route answers `404` with the reason as its error code (an `<img>` cannot
  read a JSON envelope). The GUI falls back to the exact full-text view with
  a muted reason line — fallbacks over defer-and-warn.
- **The web side is a label swap, not a new surface**: eligible chunks retitle
  the existing collapsible to "Show preview"; mounting the panel drives the
  locate fetch (closed panels unmount), TanStack Query caches it at
  `staleTime: Infinity` (content-addressed — a located page can never go
  stale), and the browser caches the PNG under the immutable header. History
  works retroactively: snapshots don't store `doc_id`, but `chunk_id` embeds
  it (`{doc_id}::{index}`), and the snapshot's `content` is the locate input.
- **PDFium is not thread-safe** (pypdfium2 METADATA): every pdfium call —
  render's PIL encode included, the bitmap buffer belongs to pdfium —
  serializes behind one module-level lock shared by locate and render.

## Rationale

- **Against ingest-time provenance** (the ADR-005 §5 seam): it requires a
  corpus-wide reingest, touches all five chunkers (`start_index` plumbing)
  plus the loader's page maps, and still cannot *highlight* — page numbers
  are not rectangles. Locate-at-preview-time needed zero schema changes, zero
  reingest, and works for every already-ingested corpus and every historical
  conversation. The seam stays open; nothing here forecloses it.
- **Against client-side PDF.js**: react-pdf's `customTextRenderer` cannot
  highlight across text items (upstream #975/#1622), the worker story under
  the current Next.js is unresolved friction, it drags ~2 MB of JS into the
  bundle, and PPTX would *still* need server conversion. Server rects are
  deterministic and format-agnostic — one code path for both formats, and the
  client stays a dumb `<img>` plus percentage-positioned `<div>`s.
- **Validated empirically before building** (2026-07-16 prototype against
  live ingested chunks): page selection by word coverage won with wide
  margins (saltmere 0.72 vs 0.57 runner-up; syllabus 1.00 vs 0.59), and
  pdfium's search tolerates line breaks, so snippet search + rect merge
  needed no fuzzy matching — even on chunks with headings, hyperlinks, and a
  GFM table with dot leaders. Live-stack checks post-build: saltmere chunk →
  page 2/2 at coverage 0.667 (13 rects), syllabus chunk → page 3/10 at
  coverage 1.0 (28 rects), overlays landing exactly on the chunks' text.
- **Prior art agrees on runtime matching**: Kotaemon (PDF.js + client-side
  LCS fuzzy matching at threshold 0.5), Onyx, and AnythingLLM all locate text
  at view time — none store boxes at ingest. Kotaemon's is the flakier half
  of the trade (fuzzy client matching over a text layer); server-side pdfium
  search is the deterministic half.
- **LibreOffice in-image over a sidecar**: single-user posture, one
  conversion per deck per container lifetime (0.5–2 s warm, ~2–12 s on the
  first cold call), and docling's own PPTX backend renders charts through the
  identical LibreOffice→PDF→pypdfium2 path — the dependency is already
  ideologically in the stack. A sidecar buys isolation nobody needs here.

## Consequences

- **No migration, no reingest, retroactive history** — including previews for
  conversations that predate the feature. Documents edited on disk since
  ingest answer `file_changed` until reingested: honest, by design.
- **Scope is digital PDF + PPTX.** Scanned PDFs keep the OCR badge + full
  text (owner scope cut — OCR'd pages have no reliable text layer to search).
  `.docx`/`.html` have no pagination to preview (converting would *invent*
  one); `.xlsx`/`.md`/`.txt` keep full text. `libreoffice-writer`/`-calc` are
  deliberately not installed.
- **The pdfium lock serializes previews.** Theoretical contention for one
  user; the recorded escape hatches if it ever matters are a per-process
  render worker or rect caching (not built). No Prometheus metrics on this
  path yet — it can ride a later observability pass.
- **The conversion cache is container-ephemeral** (`/tmp/varagity-preview/`):
  a restart re-pays one soffice run per deck; `doc_id` content-addressing
  makes staleness structurally impossible. Host-mode API runs without
  LibreOffice lose only PPTX previews (`conversion_unavailable`), never crash.
- **CJK decks render substituted glyph boxes** — `fonts-noto-cjk` (~+300 MB)
  is deliberately omitted for the English corpora this stack serves.
- **`PREVIEW_*` settings are env-only** (not in `PATCH /api/settings`);
  `PREVIEW_ENABLED=false` is the rollback: the API answers `preview_disabled`
  and the GUI shows full text everywhere. `GET /api/config` surfaces the flag
  read-only so the GUI can skip eligibility entirely.
