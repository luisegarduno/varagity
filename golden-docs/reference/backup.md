# Backups

Corpus backups ([ADR-019](../adr/ADR-019-corpus-backups.md)): one zip
bundle per backup, written into the corpus folder's reserved
`varagity-exports/` so it survives `docker compose down -v` beside the
sources. The container (`bundle.json` manifest, the all-or-nothing writer,
the one-member manifest read, the folder listing with each bundle's cheap
restore verdict — the graph engine's file format included — and the full
sha256 verification), the RAG export cut from one consistent corpus
snapshot, the RAG restore that turns a bundle back into rows and a rebuilt
BM25 index with no model call, the graph export that copies the engine's
working directory verbatim under the graph service's write lock, and the
graph restore that extracts it beside the live one and swaps it in. The
HTTP side (the corpus gate, the backup runner, the routes) is in the
[HTTP API](api.md) reference, the advisory corpus lock is on the vector
store in [Stores](stores.md), the swap itself
(`GraphService.replace_workdir`) is in [GraphRAG](graph.md), and the
`rag-export` / `rag-restore` / `graph-export` / `graph-restore` flows are in
[Orchestration](pipeline.md). The format, and how a restore reads it back,
are described in [Data model](../data-model.md#backup-bundles).

::: varagity.backup

::: varagity.backup.bundle

::: varagity.backup.rag

::: varagity.backup.graph
