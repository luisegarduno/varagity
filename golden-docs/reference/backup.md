# Backups

Corpus backups: one zip bundle per backup, written into the corpus folder's
reserved `varagity-exports/` so it survives `docker compose down -v` beside
the sources. The container (`bundle.json` manifest, the all-or-nothing
writer, the one-member manifest read, the folder listing with each bundle's
cheap restore verdict, the full sha256 verification), the RAG export cut
from one consistent corpus snapshot, and the RAG restore that turns a bundle
back into rows and a rebuilt BM25 index with no model call. The HTTP side
(the corpus gate, the backup runner, the routes) is in the
[HTTP API](api.md) reference, the advisory corpus lock is on the vector
store in [Stores](stores.md), and the `rag-export` / `rag-restore` flows are
in [Orchestration](pipeline.md). The format, and how a restore reads it
back, are described in [Data model](../data-model.md#backup-bundles).

::: varagity.backup

::: varagity.backup.bundle

::: varagity.backup.rag
