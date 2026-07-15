# Varagity

Varagity is a self-hosted, GPU-accelerated **Retrieval-Augmented Generation (RAG)**
system implementing Anthropic-style **Contextual Retrieval**: documents are parsed,
chunked, situated with LLM-generated context blurbs, then indexed for hybrid
(semantic + BM25) retrieval with cross-encoder reranking and grounded, cited
answers. The primary surface is the web GUI at `:3000` — streaming chat whose
evidence panel shows *how each answer was built*, chunk by chunk — with the
terminal CLI as a peer front-end over the same Prefect flows, and a
Prometheus/Grafana pair watching the whole pipeline.

## This site

As-built documentation of the system (v1 + v2):

- **[Architecture](architecture.md)** — service topology, the two pipelines,
  the `(doc_id, original_index)` identity thread, and why Contextual Retrieval.
- **[Data model](data-model.md)** — the chunk metadata record, PostgreSQL
  schema, Elasticsearch mapping, and idempotency semantics.
- **[Pipelines](pipelines.md)** — the ingest/query/eval Prefect task graphs
  and their retry/caching posture.
- **[Runbook](runbook.md)** — clean clone to answered question, plus every
  operational gotcha (healthchecks, volumes, GPUs, OCR, Elasticsearch).
- **[HTTP API](api.md)** — the wire contract, auto-rendered from the
  checked-in OpenAPI snapshot, plus the two SSE event protocols and the
  error-envelope conventions.
- **[ADRs](adr/index.md)** — the decisions with lasting consequences and
  their why.
- **[API reference](reference/index.md)** — rendered from the package's
  Google-style docstrings via mkdocstrings.

## Where everything else lives

- **System design** — `spec.md` at the repository root: the full, forward-looking
  v1 specification this system was built from; `spec_v2.md` beside it: the v2
  design (web GUI, reranking, observability, chunkers, modalities).
- **Implementation plans** — `thoughts/shared/plans/2026-07-09-varagity-v1-vertical-slices.md`
  and `thoughts/shared/plans/2026-07-11-varagity-v2-vertical-slices.md`:
  the vertical slices each build followed, with per-phase verification notes.

!!! warning "Naming caution"
    `docs/` in this repository is the **gitignored ingest corpus** (the documents the
    RAG system indexes), *not* project documentation. Project documentation — what you
    are reading — lives in `golden-docs/`.
