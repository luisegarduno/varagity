# Backups

Corpus backups: one zip bundle per backup, written into the corpus folder's
reserved `varagity-exports/` so it survives `docker compose down -v` beside
the sources. The container (`bundle.json` manifest, the all-or-nothing
writer, the one-member manifest read, the folder listing) and the RAG export
cut from one consistent corpus snapshot. The HTTP side (the corpus gate, the
backup runner, the routes) is in the [HTTP API](api.md) reference, and the
`rag-export` flow is in [Orchestration](pipeline.md). The format is
described in [Data model](../data-model.md#backup-bundles).

::: varagity.backup

::: varagity.backup.bundle

::: varagity.backup.rag
