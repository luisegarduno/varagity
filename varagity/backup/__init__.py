"""Corpus backups: one zip bundle per backup, in each corpus's reserved folder (ADR-019).

``docker compose down -v`` deletes every processed byte — the chunk rows,
their contextual blurbs and embeddings, the BM25 index — while the source
folders, being host binds, survive. A bundle written into
``<corpus root>/varagity-exports/`` survives with them, and holds what a
rebuild would otherwise have to recompute with the LLM and the embedder.

* :mod:`varagity.backup.bundle` — the container: the ``bundle.json``
  manifest, the all-or-nothing writer (``.partial`` → ``os.replace``, a
  sha256 per member), the one-member manifest read, and the folder listing.
* :mod:`varagity.backup.rag` — what a RAG bundle holds and how it is cut
  from one consistent corpus snapshot.

The package knows nothing about HTTP: the API's backup runner and routes
(:mod:`varagity.api.backup_runner`, :mod:`varagity.api.routes.backups`)
drive it through the ``rag-export`` Prefect flow.
"""

from varagity.backup.bundle import (
    BUNDLE_FORMAT,
    BUNDLE_VERSION,
    MANIFEST_NAME,
    BundleError,
    BundleInvalid,
    BundleListing,
    BundleManifest,
    BundleMember,
    BundleWriter,
    Corpus,
    WrittenBundle,
    bundle_name,
    corpus_root,
    display_path,
    exports_dir,
    is_bundle_name,
    list_bundles,
    read_manifest,
    write_bundle,
)
from varagity.backup.rag import NothingToExport, RagExport, export_rag

__all__ = [
    "BUNDLE_FORMAT",
    "BUNDLE_VERSION",
    "MANIFEST_NAME",
    "BundleError",
    "BundleInvalid",
    "BundleListing",
    "BundleManifest",
    "BundleMember",
    "BundleWriter",
    "Corpus",
    "NothingToExport",
    "RagExport",
    "WrittenBundle",
    "bundle_name",
    "corpus_root",
    "display_path",
    "export_rag",
    "exports_dir",
    "is_bundle_name",
    "list_bundles",
    "read_manifest",
    "write_bundle",
]
