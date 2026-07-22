# ADR-009: Office/web modalities via a generalized Docling core

**Status:** Accepted (2026-07-14)

## Context

v1 parsed `pdf` and `text`. spec_v2 §8 named `.docx/.pptx/.xlsx/.html` the
lowest-effort, highest-surface win: Docling — already in the image for PDF
([ADR-003 §5](ADR-003-vertical-build-and-ops-choices.md)) — converts all of
them natively, and the parser registry was built to grow by one file per
modality. The open questions were how much of the PDF parser generalizes,
and what page provenance each format can honestly claim.

## Decision

- **Extract the format-agnostic machinery into
  `parsers/docling_base.py`**: `DocumentConverter` conversion →
  structure-aware markdown export → hyphen repair → per-page character
  counts → `RawDocument` provenance assembly. `pdf.py` keeps its two-pass
  OCR fallback (unchanged, ADR-003 §5) but now shares the core.
- **Two thin registry parsers**: `office.py` (`@register("office")`,
  `.docx/.pptx/.xlsx`) and `web.py` (`@register("web")`, `.html/.htm`) —
  single no-OCR conversions, because these formats carry **digital text by
  construction**; the OCR fallback stays PDF-only.
- **Provenance per format (as-built, verified):** `.pptx` slides and
  `.xlsx` sheets **are Docling pages**, so slide/sheet identity rides the
  existing `page` field — no schema addition. `.docx` and `.html` expose no
  pagination (`document.pages` is empty), so `page = None`, the same
  graceful degradation as `.txt`/`.md`. Sheet-**name** provenance is
  deliberately out of scope (the plan sketched sheet identity in metadata;
  the sheet *number* in `page` covers retrieval provenance without new
  Docling metadata plumbing).
- **No new dependency**: the pinned Docling install already bundles the
  office/web converter backends (`python-docx`/`python-pptx`/`openpyxl`/
  `beautifulsoup4` — lightweight, no layout-model downloads), verified
  importable with no extra system packages.

## Rationale

- **One conversion pipeline, not five.** Hand-rolled per-format parsers
  (python-docx / pandoc / BeautifulSoup stacks) would fork the
  markdown/table/provenance shape that ADR-003 §5 deliberately unified —
  every downstream consumer (chunkers, blurbs, the evidence panel's format
  badge and `page`) works on the new formats *because* they emerge from the
  same core.
- **The registry promise held, measurably**: each modality is one file plus
  one import line; the loader's entire diff was routing the new buckets to
  their parsers. e2e proves each format is retrievable and answerable with
  format-true `file_type`/`page`/`extraction` metadata (live check: pptx
  `page=1` slide, xlsx `page=1` sheet, docx/html `page NULL`).
- **Skipping OCR for office/web is a correctness call, not a shortcut** —
  there is no raster layer to recover; running the trigger heuristics would
  add cost and a failure mode for zero recall.

## Consequences

- `ALLOWED_EXTENSIONS` widens to
  `.pdf,.txt,.md,.docx,.pptx,.xlsx,.html,.htm`; discovery gains `office`
  and `web` buckets (an allowed extension with no bucket is still warned
  and dropped).
- The GUI needed zero edits: the evidence panel's format badge and
  slide/`page` display, the upload whitelist, and `GET /api/config` all
  picked the new formats up from server truth.
- **Deferred, seams noted** (spec_v2 §15): image (`.jpg/.png`, re-adding
  llama.cpp `--mmproj`) and audio (ASR) modalities; VLM/GPU-served OCR (the
  docling-serve tier) remains the escalation if scanned volume grows —
  ADR-003 §5's post-v1 note stands, still deferred through v2.
- Anything Docling can't page (`.docx`, `.html`) stays document-level
  provenance; if per-chunk page/char offsets ever land (the deferred
  in-browser preview, [ADR-005 §5](ADR-005-web-stack-and-api.md)), this
  core's per-page character counts are where they attach.

## Amendment (2026-07-22): the low-cost widening + the `image` bucket

The registry design above was built so that new formats cost one routing
line — this amendment cashed that in, plus one genuinely new parser.

- **Formats that ride existing parsers, zero new conversion code.**
  `ALLOWED_EXTENSIONS` (still the single whitelist; the value in
  `## Consequences` above is superseded) now also admits: `.rst` → `text`;
  `.xhtml` → `web`; and, into `office`, the OOXML macro/template variants
  (`.docm`/`.dotx`/`.dotm`/`.pptm`/`.potx`/`.potm`/`.ppsx`/`.ppsm`/`.xlsm`
  — Docling's backends open them like their base formats), single-table
  `.csv`, and OpenDocument `.odt`/`.ods`/`.odp`. OpenDocument needed one
  new direct dependency (`odfdo` — Docling's backend imports it lazily);
  everything else was routing. Verified page semantics: `.ods` sheets are
  Docling pages (like `.xlsx`), `.odp` slides expose no item provenance
  (unlike `.pptx`) so `page` stays `NULL`.
- **A fifth bucket: `image` (`.png`/`.jpg`/`.jpeg`/`.tif`/`.tiff`/`.bmp`/
  `.webp`), OCR-only.** Bitmaps have no text layer, so the two-pass trigger
  logic is meaningless — `parsers/image.py` reuses the PDF parser's
  OCR-engine factory and always converts with the configured engine in
  full-page mode. Its chunks carry a third `extraction` value, `"ocr"`
  (deliberately distinct from `"ocr_fallback"`: OCR is this format's only
  path, not a recovery), which the GUI's OCR badge and extraction-mix
  column also recognize. This ships the *text* half of the deferred image
  modality; semantic image understanding (re-adding llama.cpp `--mmproj`)
  stays deferred as noted above.
- **The GUI consequence held again**: accept-lists, captions, and format
  badges picked the new formats up from server truth; the only frontend
  edits were widening the two OCR-badge checks to the new value.
