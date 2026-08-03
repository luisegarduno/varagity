# GraphRAG

The graph subsystem (spec_graphrag §5.2, §10;
[ADR-017](../adr/ADR-017-graphrag-engine.md)) — a **peer** retrieval
corpus beside chunk-RAG, not a sixth retriever: structured messages (an
iMessage `chat.db` in v1) become a knowledge graph with per-message
provenance.

Stage 1 ships the engine-independent foundations plus the bake-off
infrastructure that decided the engine; the graph build pipeline, query
path, and web surfaces are stage 2.

- **Message sources** are a registry family chosen by **structural
  dispatch** — a file goes to the first source whose `matches()` accepts
  it (the parser precedent), so there is no `config.py` vocabulary tuple
  to keep in lockstep. v1 registers only `imessage`.
- **Graph engines** are a registry family in the usual `@register("name")`
  shape. The three registered here are the ADR-017 bake-off seats; the
  losers are deleted in stage 2 and the seam is what stays.
  **Importing the package must not import an engine library** — each
  adapter imports its heavy dependency inside `session()`, which is why
  CI never installs the `bakeoff` dependency group.
- **Rendering** is pure and shared: the same messages become thread-day
  transcripts for document-shaped engines and one episode per message for
  episode-shaped ones, so the bake-off compares engines rather than diets.

## Message sources

::: varagity.graph.sources.base

::: varagity.graph.sources.imessage

## Records, engine seam, and rendering

::: varagity.graph.records

::: varagity.graph.base

::: varagity.graph.render

## Bake-off adapters

::: varagity.graph.engines.lightrag

::: varagity.graph.engines.cognee

::: varagity.graph.engines.graphiti
