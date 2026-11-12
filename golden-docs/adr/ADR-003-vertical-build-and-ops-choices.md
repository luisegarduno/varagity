# ADR-003: Vertical-slice build order and v1 operational choices

**Status:** Accepted (implementation plan, 2026-07-09) · Amended (2026-11-11 — [below](#amendment-2026-11-11-the-served-tokenizer-fits-the-preamble))

One record for the cluster of build/ops decisions made when the v1
implementation plan was drawn up. Each was small alone; together they shaped
how the system was built and runs.

## 1. Build vertically, not horizontally

**Context.** Spec §19.2 sketched a horizontal order: build every component,
wire them at the end. Nothing would have run end-to-end until step ~10.

**Decision.** Build in ten vertical slices: a working Q&A system by the fourth slice
(vanilla RAG), then one independently-testable capability per slice
(contextual embeddings → contextual BM25 → PDF/OCR → Prefect → eval →
hardening), with services and dependencies added only in the slice that
first uses them.

**Consequences.** `docker compose up` was never broken; every infra addition
was smoke-tested by code using it; the Anthropic quality ladder mapped 1:1
onto slices and the eval measures exactly those configurations.
Prefect arrived late as thin adapters over functions built in the earlier
slices — the `IngestStages` seam keeps one orchestration loop for both plain
and tracked execution.

## 2. Prefect server on its default SQLite

**Context.** The official production Prefect compose runs Postgres + Redis.

**Decision.** v1 backs the Prefect server with its default SQLite in the
`prefect` volume; flows run in-process from the CLI — no workers,
deployments, or schedules.

**Consequences.** One fewer stateful service to operate; run history resets
with `down -v`. Fine for a single-user dev stack; a production posture would
revisit both the backing store and the deployment model.

## 3. Eval isolation via ephemeral testcontainers

**Context.** The eval matrix must ingest the corpus under multiple
configurations (contextual on/off, per OCR engine) without polluting or
wiping the live stores.

**Decision.** `main.py eval` spins throwaway Postgres/Elasticsearch
containers per run (`varagity/eval/containers.py`, shared with the
integration/e2e test fixtures) and uses the live GPU services — which are
stateless — for embeddings/LLM.

**Consequences.** Eval is safe to run any time; results depend only on the
pinned eval settings and the corpus. The ephemeral ES disables disk-watermark
allocation checks (throwaway stores must never depend on host disk pressure —
a real incident; see the
[runbook](../runbook.md#elasticsearch-notes)). Docker is required for eval.

## 4. `doc_id` hashes the *relative* path

**Context.** Spec §8.1 said `doc_id = hash(absolute path + content hash)`,
but absolute paths differ between host (`/home/…/docs/a.md`) and container
(`/app/docs/a.md`) and across machines.

**Decision.** Hash the path **relative to `DOCS_PATH`** (plus the byte
hash), keep the absolute path only as `source` provenance.

**Consequences.** Idempotency holds across host/container runs; golden eval
refs resolve to `chunk_id`s from corpus files alone (no store round-trip);
eval sets are portable. Moving a file within the corpus changes its
`doc_id` (treated as a new document — acceptable for v1).

## 5. Two-pass PDF extraction with the OCR fallback inside Docling

**Context.** OCR is a multi-× slowdown that most (digital) PDFs don't need;
the reference notebook fell back to a separate pdf2image + pytesseract path,
which produces differently-shaped output than the primary parser.

**Decision.** Pass 1 converts with `do_ocr=False`; a per-document trigger
(near-empty text, high textless-page ratio, or a pass-1 exception)
re-converts with `do_ocr=True` — **staying inside Docling** for both passes,
with the engine pluggable via an `OCR_ENGINE` factory
(EasyOCR/Tesseract; engine choice is [ADR-004](ADR-004-ocr-engine-choice.md)).

**Consequences.** Both passes share one structure-aware markdown/table/page
pipeline, so the fallback changes *how* text is recovered, never its
downstream shape; only textless documents pay the OCR cost; chunks carry
`extraction` provenance. `PDF_OCR_FORCE_FULL_PAGE` exists because a
*corrupt* text layer passes content triggers by definition. GPU/VLM OCR
serving remains the post-v1 escalation if scanned volume grows. *(v2
update: still deferred — v2 expanded modalities to office/web digital text
instead; see [ADR-009](ADR-009-modality-expansion.md).)*

## 6. Smaller calls recorded with their rationale

- **Skeleton-first identity trick:** the `contextualized_content NOT NULL`
  column was satisfied pre-contextualization by `content` itself — no schema
  migration when contextual embeddings landed.
- **`CONTEXTUALIZE` stays a knob**, not a constant: it is the eval baseline
  (config 1) and a throughput escape hatch.
- **CLI = stdlib argparse** — the spec named no CLI framework and `rich`
  owns presentation; no dependency earns its keep here.
- **`n_tokens` via tiktoken cl100k as a documented approximation** — the e5
  tokenizer differs; the count feeds provenance and a ≥480-token warning
  (e5 truncates at 512), not billing.
- **Unique index on `(doc_id, original_index)`** — the fusion identity is
  load-bearing; corrupt it loudly at write time, not silently at query time.
- **Coverage floor 80%**, ratcheting up as the suite grows.

## Amendment (2026-11-11): the served tokenizer fits the preamble

The cl100k approximation above had also been sizing one thing it cannot
see: the contextualizer's document preamble against the llama.cpp window.
That is a choice of *content*, and on digit-dense text cl100k misses by
more than the window's slack.

- **The failure.** `Snapchat/html/location_history.html` (186k
  characters, 27% digits: timestamps and coordinates) failed
  contextualization on every ingest from August on. The served model
  (Qwythos-9B on llama.cpp) gives every digit its own token where cl100k
  packs up to three, so the cl100k-sized preamble — the full 11,648-token
  budget — counted **16,921** on the server (1.45×). Every chunk's prompt
  (17,084 tokens) drew a 400 `exceed_context_size_error` from the
  16,384-token window. A failed file is non-fatal by design, so it stayed
  out of the corpus with only the logs to say so.
- **Decision: hold the preamble to the served model's own count.**
  `LLMClient.count_served_tokens()` asks llama.cpp's `/tokenize` (at the
  server root — `/v1/tokenize` is a 404). `fit_document_preamble` runs
  once per document. A document of at most budget-many UTF-8 bytes is
  taken whole, uncounted, because every token spans at least one byte.
  Otherwise cl100k cuts first (cheap, and it bounds the request), then the
  served count checks the cut and re-cuts it in proportion while it is
  over. One re-cut landed the export at 11,363 served tokens: two
  `/tokenize` calls, ~25 ms each. The budget arithmetic is unchanged;
  its unit is now the served token.
- **Once per document, never per chunk.** The preamble must stay
  byte-identical across a document's chunks (llama.cpp's prompt cache,
  spec §9.4), so `situate_context` sends the fitted preamble as given.
  Re-fitting per chunk would re-pay the count hundreds of times, and a
  count failing mid-document would hand chunks different preambles.
- **No served count → the byte bound.** If `/tokenize` stays
  unreachable through its retries, or the backend is not llama.cpp (a
  4xx latches counting off for that client), the preamble is the
  document's first budget-many UTF-8 bytes: a fit under any tokenizer.
  The cost is a shorter context, never a failed file, and a warning
  says so.
- **`_fit_max_tokens` keeps cl100k, deliberately.** Measured on the
  served build (b9935, `--parallel 1`): a generation that reaches the
  window stops there with `finish_reason=length`, no mid-decode 500.
  And a prompt that alone overflows fails either way, as the client's
  `ValueError` or the server's 400 (both reach the chat stream as the same
  `pipeline_error`). An exact count there would change no outcome and add a
  round trip to every chat call, so it is spent only where it changes what
  is sent.
- **Consequences.** Of the 16 corpus documents only the export is cut;
  the served model counts this corpus's prose 2–9% over cl100k, and its
  digit-dense contact list 1.80×, but both stay inside the budget whole.
  A prose document *over* the budget now gets a preamble a few percent
  shorter than before: the 640-token overhead reserve was sized to absorb
  that drift and is now mostly slack. The graph paths still budget the
  window by approximation (the LightRAG adapter's `fit_max_tokens`: cl100k;
  the graph answer context: 3 characters per token), so digit-dense
  transcripts are their open risk.
