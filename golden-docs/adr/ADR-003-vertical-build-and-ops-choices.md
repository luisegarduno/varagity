# ADR-003: Vertical-slice build order and v1 operational choices

**Status:** Accepted (implementation plan, 2026-07-09)

One record for the cluster of build/ops decisions made when the v1
implementation plan was drawn up. Each was small alone; together they shaped
how the system was built and runs.

## 1. Build vertically, not horizontally

**Context.** Spec §19.2 sketched a horizontal order: build every component,
wire them at the end. Nothing would have run end-to-end until step ~10.

**Decision.** Build in ten vertical slices: a working Q&A system by Phase 4
(vanilla RAG), then one independently-testable capability per phase
(contextual embeddings → contextual BM25 → PDF/OCR → Prefect → eval →
hardening), with services and dependencies added only in the phase that
first uses them.

**Consequences.** `docker compose up` was never broken; every infra addition
was smoke-tested by code using it; the Anthropic quality ladder mapped 1:1
onto phases and the Phase 9 eval measures exactly those configurations.
Prefect arrived in Phase 8 as thin adapters over functions built in Phases
3–7 — the `IngestStages` seam keeps one orchestration loop for both plain
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
serving remains the post-v1 escalation if scanned volume grows.

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
- **Coverage floor 80%** (Phase 10), ratcheting up as the suite grows.
