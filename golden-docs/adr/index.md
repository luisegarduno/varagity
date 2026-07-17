# Architecture Decision Records

Decisions with lasting consequences, recorded so the *why* survives the
conversation that produced it. Format: Context → Decision → Consequences.

| ADR | Decision | Status |
|---|---|---|
| [ADR-001](ADR-001-pgvector-over-qdrant.md) | PostgreSQL + pgvector as the vector store (Qdrant-GPU dropped) | Accepted |
| [ADR-002](ADR-002-infinity-over-fastembed.md) | infinity as the embedding service (FastEmbed dropped) | Accepted |
| [ADR-003](ADR-003-vertical-build-and-ops-choices.md) | Vertical-slice build order + v1 operational choices | Accepted |
| [ADR-004](ADR-004-ocr-engine-choice.md) | EasyOCR as the shipped OCR fallback engine (benchmark-decided) | Accepted |
| [ADR-005](ADR-005-web-stack-and-api.md) | The v2 web GUI + HTTP API stack (Next.js + FastAPI + SSE; single-user) | Accepted |
| [ADR-006](ADR-006-reranking-wired.md) | Reranking wired into the query path as a composing retriever (≈67% tier) | Accepted |
| [ADR-007](ADR-007-observability-stack.md) | In-app Prometheus metrics + provisioned Grafana (exporters optional) | Accepted |
| [ADR-008](ADR-008-chunking-default.md) | `recursive_character` stays the chunking default (benchmark-decided) | Accepted |
| [ADR-009](ADR-009-modality-expansion.md) | Office/web modalities via a generalized Docling core (images/audio deferred) | Accepted |
| [ADR-010](ADR-010-document-page-preview.md) | Evidence-panel page previews via on-demand server-side locate + render | Accepted |
