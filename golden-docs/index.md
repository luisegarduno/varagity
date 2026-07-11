# Varagity

Varagity is a self-hosted, GPU-accelerated **Retrieval-Augmented Generation (RAG)**
system implementing Anthropic-style **Contextual Retrieval**: documents are parsed,
chunked, situated with LLM-generated context blurbs, then indexed for hybrid
(semantic + BM25) retrieval and grounded, cited terminal Q&A.

## This site

As-built documentation of the v1 system:

- **[Architecture](architecture.md)** — service topology, the two pipelines,
  the `(doc_id, original_index)` identity thread, and why Contextual Retrieval.
- **[Data model](data-model.md)** — the chunk metadata record, PostgreSQL
  schema, Elasticsearch mapping, and idempotency semantics.
- **[Pipelines](pipelines.md)** — the ingest/query/eval Prefect task graphs
  and their retry/caching posture.
- **[Runbook](runbook.md)** — clean clone to answered question, plus every
  operational gotcha (healthchecks, volumes, GPUs, OCR, Elasticsearch).
- **[ADRs](adr/index.md)** — the decisions with lasting consequences and
  their why.
- **[API reference](reference/index.md)** — rendered from the package's
  Google-style docstrings via mkdocstrings.

## Where everything else lives

- **System design** — `spec.md` at the repository root: the full, forward-looking
  v1 specification this system was built from.
- **Implementation plan** — `thoughts/shared/plans/2026-07-09-varagity-v1-vertical-slices.md`:
  the ten vertical slices the build followed, with per-phase verification notes.

!!! warning "Naming caution"
    `docs/` in this repository is the **gitignored ingest corpus** (the documents the
    RAG system indexes), *not* project documentation. Project documentation — what you
    are reading — lives in `golden-docs/`.
