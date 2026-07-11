# Architecture Decision Records

Decisions with lasting consequences, recorded so the *why* survives the
conversation that produced it. Format: Context → Decision → Consequences.

| ADR | Decision | Status |
|---|---|---|
| [ADR-001](ADR-001-pgvector-over-qdrant.md) | PostgreSQL + pgvector as the vector store (Qdrant-GPU dropped) | Accepted |
| [ADR-002](ADR-002-infinity-over-fastembed.md) | infinity as the embedding service (FastEmbed dropped) | Accepted |
| [ADR-003](ADR-003-vertical-build-and-ops-choices.md) | Vertical-slice build order + v1 operational choices | Accepted |
| [ADR-004](ADR-004-ocr-engine-choice.md) | EasyOCR as the shipped OCR fallback engine (benchmark-decided) | Accepted |
