# ADR-004: EasyOCR as the shipped OCR fallback engine

**Status:** Accepted (decided by the Phase 9 benchmark, 2026-07-11)

## Context

The PDF parser's OCR fallback ([ADR-003 §5](ADR-003-vertical-build-and-ops-choices.md))
made the engine pluggable (`OCR_ENGINE` → Docling `ocr_options` factory) and
shipped **EasyOCR as an explicitly provisional default** — the plan required
the shipped default to be decided by benchmark data, not preference. The
`eval ocr` harness measures, per engine: **CER/WER** against the OCR
fixtures' known ground truth (`data/eval/ocr_truth/`), **pages/sec**
wall-clock (warmed engines, CPU), and — the project-relevant definition of
"better" — **retrieval recall** on the scanned-document golden queries with
the engine as the only variable (`CONTEXTUALIZE` off).

## Benchmark results

Run `20260711T172924Z-ocr.json` (3 fixture pages: a 1-page scanned memo, a
2-page mixed digital/scanned survey; 4 scanned-doc golden queries):

| Engine | CER | WER | Pages/s | recall@{5,10,20} (all 3 methods) |
|---|---|---|---|---|
| **easyocr** | **0.0000** | **0.0000** | 0.102 | 1.000 |
| tesseract | 0.0014 | 0.0161 | **0.551** | 1.000 |

- Tesseract's errors are concentrated on the scanned memo (WER 0.0526 there —
  two dropped/garbled words); it was error-free on the mixed survey.
- Both engines' chunk boundaries resolved every golden ref (no drift).
- Retrieval recall saturates at 1.000 for both — on this tiny corpus the
  engine choice is **not** a retrieval-quality signal either way.

## Decision

**EasyOCR ships as the default** (`OCR_ENGINE=easyocr`). Tesseract remains
one config flip away.

## Rationale

The data reduces this to a quality-vs-throughput call, and v1's usage profile
weights quality:

- **The fallback's whole purpose is recovering text for retrieval**, and OCR
  noise hits BM25 keyword matching hardest — a dropped word is a permanently
  unmatchable keyword. Tesseract produced real word errors on exactly the
  kind of low-fidelity scan the fallback exists for; EasyOCR was error-free.
- **Throughput is the wrong thing to optimize here**: only textless documents
  trigger OCR, ingestion is offline batch work, and 5.4× faster
  (0.55 vs 0.10 pages/s) only matters at a scanned-document volume v1 doesn't
  have. If that volume arrives, the plan's escalation path is GPU/VLM OCR
  serving — not a lossier CPU engine.
- **Zero migration cost**: EasyOCR was already the provisional default and
  its weights are already in the cache volume.

## Consequences

- `OCR_ENGINE=easyocr` stays the default in `config.py` / `.env.example`,
  now marked as benchmark-decided rather than provisional.
- Set `OCR_ENGINE=tesseract` when throughput matters more than fidelity for
  a specific batch (e.g. a large, high-quality scan backlog) — the Tesseract
  system binary ships in the app image, so it always works.
- The benchmark is re-runnable (`uv run --group eval main.py eval ocr`);
  revisit this decision if the fixture set grows toward real-world scan
  quality or a new engine (e.g. RapidOCR — one factory line) is added.
