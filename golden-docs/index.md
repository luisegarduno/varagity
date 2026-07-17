---
hide:
  - navigation
  - toc
---

# Varagity

Varagity is a self-hosted, GPU-accelerated **Retrieval-Augmented Generation (RAG)**
system implementing Anthropic-style **Contextual Retrieval**: documents are parsed,
chunked, situated with LLM-generated context blurbs, then indexed for hybrid
(semantic + BM25) retrieval with cross-encoder reranking and grounded, cited
answers. The primary surface is the web GUI at `:3000` — streaming chat whose
evidence panel shows *how each answer was built*, chunk by chunk — with the
terminal CLI as a peer front-end over the same Prefect flows, and a
Prometheus/Grafana pair watching the whole pipeline.

This site is the **as-built documentation** of the system (v1 + v2):

<div class="grid cards" markdown>

-   :material-graph:{ .lg .middle } __Architecture__

    ---

    Service topology, the two pipelines, the `(doc_id, original_index)`
    identity thread, and why Contextual Retrieval.

    [:octicons-arrow-right-24: Read the architecture](architecture.md)

-   :material-database-search:{ .lg .middle } __Data model__

    ---

    The chunk metadata record, PostgreSQL schema, Elasticsearch mapping,
    and idempotency semantics.

    [:octicons-arrow-right-24: Read the data model](data-model.md)

-   :material-source-branch:{ .lg .middle } __Pipelines__

    ---

    The ingest/query/eval Prefect task graphs and their retry/caching
    posture.

    [:octicons-arrow-right-24: Follow the pipelines](pipelines.md)

-   :material-wrench:{ .lg .middle } __Runbook__

    ---

    Clean clone to answered question, plus every operational gotcha —
    healthchecks, volumes, GPUs, OCR, Elasticsearch.

    [:octicons-arrow-right-24: Open the runbook](runbook.md)

-   :material-api:{ .lg .middle } __HTTP API__

    ---

    The wire contract, auto-rendered from the checked-in OpenAPI snapshot,
    plus the two SSE event protocols and the error-envelope conventions.

    [:octicons-arrow-right-24: Browse the API](api.md)

-   :material-scale-balance:{ .lg .middle } __ADRs__

    ---

    The decisions with lasting consequences and their why — from
    pgvector-over-Qdrant to document page previews.

    [:octicons-arrow-right-24: Read the decisions](adr/index.md)

-   :material-book-open-variant:{ .lg .middle } __API reference__

    ---

    Every public module, class, and function, rendered from the package's
    Google-style docstrings via mkdocstrings.

    [:octicons-arrow-right-24: Explore the reference](reference/index.md)

-   :material-file-document-edit:{ .lg .middle } __Specs & plans__

    ---

    `spec.md` and `spec_v2.md` at the repository root are the forward-looking
    designs this system was built from; `thoughts/shared/plans/` holds the
    vertical slices each build followed.

    [:octicons-arrow-right-24: Visit the repository](https://github.com/luisegarduno/varagity)

</div>

!!! warning "Naming caution"
    `docs/` in this repository is the **gitignored ingest corpus** (the documents the
    RAG system indexes), *not* project documentation. Project documentation — what you
    are reading — lives in `golden-docs/`.
