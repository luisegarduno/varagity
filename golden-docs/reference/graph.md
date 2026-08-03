# GraphRAG

The graph subsystem (spec_graphrag §5.2, §10;
[ADR-017](../adr/ADR-017-graphrag-engine.md)) — a **peer** retrieval
corpus beside chunk-RAG, not a sixth retriever: structured messages (an
iMessage `chat.db` in v1) become a knowledge graph with per-message
provenance.

Stage 1 shipped the engine-independent foundations plus the bake-off
harness that decided the engine; stage 2 productionizes the winner. The
API routes, query path, and web surfaces are the remaining phases.

- **Message sources** are a registry family chosen by **structural
  dispatch** — a file goes to the first source whose `matches()` accepts
  it (the parser precedent), so there is no `config.py` vocabulary tuple
  to keep in lockstep. v1 registers only `imessage`.
- **Graph engines** are a registry family in the usual `@register("name")`
  shape, selected by `GRAPH_ENGINE`. ADR-017 picked `lightrag` and the
  losing adapters were deleted at stage-2 start; the seam is what stays,
  so the decision remains benchmark-revisitable.
  **Importing the package must not import an engine library** — the
  adapter imports `lightrag` inside `session()`, so the registry stays
  free even though the engine is a main dependency.
- **Rendering** is pure and shared: messages become thread-day
  transcripts, and identical thread-days always render to an identical
  `doc_key` and text — the property the build diff rests on.
- **The workdir owns two sidecars.** `varagity_manifest.json` records
  `doc_key → {content hash, message guids, thread, span}`, which makes a
  build a **diffing upsert** (LightRAG's enqueue dedup would otherwise
  keep a stale transcript forever) and keeps provenance durable across
  restarts. `varagity_graph_summary.json` caches the graph's size so
  status polls and metric scrapes never walk the graphml.
- **One process is the single writer.** `varagity.graph.service` holds
  the lone session, guards writes with a single-flight lock, and lets
  reads through unlocked — the session runs its event loop on its own
  thread, so a chat query is answered while a multi-day backfill is still
  extracting.
- **Answer synthesis is the repo's, not the engine's** (ADR-017 chose
  retrieval-only): the adapter returns entities, relations, and transcript
  excerpts, and `varagity.graph.answer` writes the grounded, capped,
  `<think>`-stripped answer over them.

## Message sources

::: varagity.graph.sources.base

::: varagity.graph.sources.imessage

## Records, engine seam, rendering, and synthesis

::: varagity.graph.records

::: varagity.graph.base

::: varagity.graph.render

::: varagity.graph.answer

## Workdir state and the process-wide handle

::: varagity.graph.manifest

::: varagity.graph.service

## Engine adapter

::: varagity.graph.engines.lightrag
