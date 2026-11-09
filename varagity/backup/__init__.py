"""Corpus backups: one zip bundle per backup, in each corpus's reserved folder (ADR-019).

``docker compose down -v`` deletes every processed byte — the chunk rows,
their contextual blurbs and embeddings, the BM25 index, the extracted
message graph — while the source folders, being host binds, survive. A
bundle written into ``<corpus root>/varagity-exports/`` survives with them,
and holds what a rebuild would otherwise have to recompute with the LLM and
the embedder.

* :mod:`varagity.backup.bundle` — the container: the ``bundle.json``
  manifest, the all-or-nothing writer (``.partial`` → ``os.replace``, a
  sha256 per member), the one-member manifest read, the folder listing with
  each bundle's cheap restore verdict, and the full sha256 verification.
* :mod:`varagity.backup.rag` — what a RAG bundle holds, how it is cut from
  one consistent corpus snapshot, and how a restore turns it back into rows
  (and a rebuilt BM25 index) with no model call.
* :mod:`varagity.backup.graph` — what a graph bundle holds (the engine's
  working directory, verbatim) and how it is copied under the graph
  service's write lock while chat keeps answering.

The package knows nothing about HTTP: the API's backup runner and routes
(:mod:`varagity.api.backup_runner`, :mod:`varagity.api.routes.backups`)
drive it through the ``rag-export``, ``rag-restore`` and ``graph-export``
Prefect flows.
"""

from varagity.backup.bundle import (
    BUNDLE_FORMAT,
    BUNDLE_VERSION,
    MANIFEST_NAME,
    BundleDamaged,
    BundleError,
    BundleIncompatible,
    BundleInvalid,
    BundleListing,
    BundleManifest,
    BundleMember,
    BundleSource,
    BundleVerdict,
    BundleWriter,
    Corpus,
    MemberStream,
    NothingToExport,
    WrittenBundle,
    bundle_name,
    corpus_root,
    display_path,
    exports_dir,
    inspect_bundle,
    is_bundle_name,
    is_graph_file_name,
    list_bundles,
    open_bundle,
    read_manifest,
    verify_bundle,
    write_bundle,
)
from varagity.backup.graph import (
    GraphExportReport,
    export_graph,
    graph_counts,
    workdir_files,
)
from varagity.backup.rag import (
    BM25RebuildFailed,
    RagExport,
    RagRestore,
    derive_corpus_stale,
    export_rag,
    ingest_drift,
    restore_rag,
    restored_source,
)

__all__ = [
    "BUNDLE_FORMAT",
    "BUNDLE_VERSION",
    "MANIFEST_NAME",
    "BM25RebuildFailed",
    "BundleDamaged",
    "BundleError",
    "BundleIncompatible",
    "BundleInvalid",
    "BundleListing",
    "BundleManifest",
    "BundleMember",
    "BundleSource",
    "BundleVerdict",
    "BundleWriter",
    "Corpus",
    "GraphExportReport",
    "MemberStream",
    "NothingToExport",
    "RagExport",
    "RagRestore",
    "WrittenBundle",
    "bundle_name",
    "corpus_root",
    "derive_corpus_stale",
    "display_path",
    "export_graph",
    "export_rag",
    "exports_dir",
    "graph_counts",
    "ingest_drift",
    "inspect_bundle",
    "is_bundle_name",
    "is_graph_file_name",
    "list_bundles",
    "open_bundle",
    "read_manifest",
    "restore_rag",
    "restored_source",
    "verify_bundle",
    "workdir_files",
    "write_bundle",
]
