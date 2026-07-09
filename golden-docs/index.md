# Varagity

Varagity is a self-hosted, GPU-accelerated **Retrieval-Augmented Generation (RAG)**
system implementing Anthropic-style **Contextual Retrieval**: documents are parsed,
chunked, situated with LLM-generated context blurbs, then indexed for hybrid
(semantic + BM25) retrieval and grounded, cited terminal Q&A.

## Where things live

- **System design** — `spec.md` at the repository root: the full v1 specification.
- **Implementation plan** — `thoughts/shared/plans/2026-07-09-varagity-v1-vertical-slices.md`:
  the vertically-sliced build order this system is being built in.
- **This site** (`golden-docs/`) — as-built architecture documentation. It grows with
  the system; the full architecture / data-model / pipelines / runbook / ADR set lands
  with the hardening phase.
- **[API reference](reference/api.md)** — rendered from the package's Google-style
  docstrings via mkdocstrings.

!!! warning "Naming caution"
    `docs/` in this repository is the **gitignored ingest corpus** (the documents the
    RAG system indexes), *not* project documentation. Project documentation — what you
    are reading — lives in `golden-docs/`.
