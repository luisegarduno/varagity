"""Prefect backup flows: the RAG export (ADR-019).

A thin ``@flow`` shell over :func:`varagity.backup.rag.export_rag`, for the
reason the graph build is one: the work belongs to the backup package (one
corpus snapshot, one all-or-nothing bundle write), while Prefect owns the
*record* that a backup ran, how long it took, and whether it failed. One
flow run per export — not a task per member: an export is a single
consistent read, and slicing it into task runs would invent boundaries the
snapshot does not have. Per-stage progress travels as the API runner's
``progress`` events instead (tracking and streaming compose, as they do for
ingest and the graph build).

No Prefect-level retries: a failed export leaves no bundle behind (the
partial is removed), so the honest response is to surface the failure and
let the user press Export again, not to re-read the corpus silently.
Parameter validation is off because the store is a live handle, not a
serializable payload.
"""

from pathlib import Path

from prefect import flow
from prefect.logging import get_run_logger

from varagity.backup.rag import ProgressCallback, RagExport, export_rag
from varagity.stores.app_settings_store import AppSettingsStore
from varagity.stores.vector_store import ContextualVectorDB


@flow(name="rag-export", validate_parameters=False)
def rag_export_flow(
    dest: Path,
    *,
    ingest_settings: dict[str, str],
    on_progress: ProgressCallback | None = None,
    store: ContextualVectorDB | None = None,
    stale: bool | None = None,
) -> RagExport:
    """Export the chunk corpus into a new bundle as a tracked flow run.

    Args:
        dest: The exports folder (``<DOCS_PATH>/varagity-exports``).
        ingest_settings: The reingest-affecting settings at export, in
            env-string form (the runner reads them before the stale flag,
            so a settings change racing the export can only make the bundle
            *more* conservative about staleness, never less).
        on_progress: Called with ``(stage, current, total)`` per stage.
        store: The vector store to snapshot; a connection of its own is
            opened (and closed) when omitted.
        stale: The corpus-stale flag at export; read from ``app_settings``
            when omitted.

    Returns:
        The published bundle and how long the export took.
    """
    logger = get_run_logger()
    if stale is None:
        with AppSettingsStore() as settings_store:
            stale = settings_store.is_corpus_stale()
    if store is not None:
        result = export_rag(dest, store, ingest_settings, stale=stale, on_progress=on_progress)
    else:
        with ContextualVectorDB() as owned:
            result = export_rag(dest, owned, ingest_settings, stale=stale, on_progress=on_progress)
    counts = result.counts
    logger.info(
        "rag export wrote %s: %d document(s), %d chunk(s), %d bytes in %.1fs",
        result.bundle.name,
        counts.get("documents", 0),
        counts.get("chunks", 0),
        result.bundle.size_bytes,
        result.wall_clock_s,
    )
    return result
