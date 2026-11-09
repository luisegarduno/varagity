"""Prefect backup flows: the RAG export and restore, the graph export (ADR-019).

Thin ``@flow`` shells over :func:`varagity.backup.rag.export_rag`,
:func:`varagity.backup.rag.restore_rag` and
:func:`varagity.backup.graph.export_graph`, for the reason the graph build is
one: the work belongs to the backup package (one corpus snapshot and one
all-or-nothing bundle write; one replacing transaction and one index
rebuild; one copy of the graph's workdir under its write lock), while
Prefect owns the *record* that a backup ran, how long it took, and whether
it failed. One flow run per operation — not a task per member or per stage:
an export is a single consistent read and a restore a single transaction,
and slicing either into task runs would invent boundaries they do not have.
Per-stage progress travels as the API runner's ``progress`` events instead
(tracking and streaming compose, as they do for ingest and the graph build).

No Prefect-level retries: a failed export leaves no bundle behind (the
partial is removed) and a failed restore leaves the previous corpus (its
transaction rolled back) — or, failing in the BM25 rebuild after the commit,
says so — so the honest response is to surface the failure and let the user
press the button again, not to repeat the work silently. Parameter
validation is off because the stores (and the graph service) are live
handles, not serializable payloads.
"""

from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

from prefect import flow
from prefect.logging import get_run_logger

from varagity.backup.graph import GraphExportProgressCallback, GraphExportReport, export_graph
from varagity.backup.rag import (
    ProgressCallback,
    RagExport,
    RagRestore,
    RestoreProgressCallback,
    derive_corpus_stale,
    export_rag,
    restore_rag,
)
from varagity.config import get_settings
from varagity.graph.service import GraphService, get_graph_service
from varagity.stores.app_settings_store import AppSettingsStore
from varagity.stores.bm25_store import ElasticsearchBM25
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


@flow(name="rag-restore", validate_parameters=False)
def rag_restore_flow(
    bundle: Path,
    *,
    ingest_settings: Callable[[], Mapping[str, str]],
    on_progress: RestoreProgressCallback | None = None,
    store: ContextualVectorDB | None = None,
    bm25: ElasticsearchBM25 | None = None,
    settings_store: AppSettingsStore | None = None,
) -> RagRestore:
    """Restore the chunk corpus from a bundle as a tracked flow run.

    Replaces both stores' corpus with the bundle's rows (re-rooted onto this
    stack's ``DOCS_PATH``) and rebuilds the BM25 index from them, then — and
    only once all of that succeeded — writes the corpus-stale flag the
    bundle implies: stale if it was stale at export, or if its rows were
    made under other reingest-affecting settings than this stack's.

    Args:
        bundle: The bundle file (inside ``<DOCS_PATH>/varagity-exports``).
        ingest_settings: Provider of this stack's reingest-affecting
            settings, in env-string form — called when the stale flag is
            written, so a settings change made *during* the restore is
            judged against the restored rows too.
        on_progress: Called with ``(stage, current, total)`` per stage
            (``validate`` → ``load`` → ``index`` → ``bm25``).
        store: The vector store to replace; a connection of its own is
            opened (and closed) when omitted.
        bm25: The BM25 store to rebuild; one is constructed (and closed)
            when omitted.
        settings_store: Where the stale flag is written; a connection of its
            own is opened (and closed) when omitted.

    Returns:
        What was restored, with the stale flag it left.
    """
    logger = get_run_logger()
    docs_root = Path(get_settings().DOCS_PATH)
    with ExitStack() as stack:
        active_store = store if store is not None else stack.enter_context(ContextualVectorDB())
        active_bm25 = bm25 if bm25 is not None else stack.enter_context(ElasticsearchBM25())
        result = restore_rag(bundle, active_store, active_bm25, docs_root, on_progress=on_progress)
    stale = derive_corpus_stale(result.manifest, ingest_settings())
    if settings_store is not None:
        settings_store.set_corpus_stale(stale)
    else:
        with AppSettingsStore() as owned_settings:
            owned_settings.set_corpus_stale(stale)
    logger.info(
        "rag restore loaded %s: %d document(s), %d chunk(s), %d in BM25 in %.1fs%s",
        result.bundle_name,
        result.documents,
        result.chunks,
        result.bm25_indexed,
        result.wall_clock_s,
        " — the corpus is flagged stale" if stale else "",
    )
    return replace(result, stale=stale)


@flow(name="graph-export", validate_parameters=False)
def graph_export_flow(
    dest: Path,
    *,
    on_progress: GraphExportProgressCallback | None = None,
    service: GraphService | None = None,
    stale: bool | None = None,
) -> GraphExportReport:
    """Export the message graph's working directory into a new bundle as a tracked flow run.

    The copy runs under the graph service's write lock, so no build can
    change a file mid-copy while graph chat keeps answering. Needs no model
    service and opens no session — only postgres, for the stale flag.

    Args:
        dest: The exports folder (``<GRAPH_DOCS_PATH>/varagity-exports``).
        on_progress: Called with ``(stage, current, total)`` per stage
            (``snapshot`` → ``pack``, counting bytes → ``finalize``).
        service: The graph service whose workdir to copy; the process's own
            (the graph's single writer) when omitted.
        stale: The graph-stale flag to record; when omitted, it is read from
            ``app_settings`` once the files are copied, so a source deleted
            mid-export is recorded.

    Returns:
        The published bundle and how long the export took.
    """
    logger = get_run_logger()

    def read_stale() -> bool:
        if stale is not None:
            return stale
        with AppSettingsStore() as settings_store:
            return settings_store.is_graph_stale()

    active = service if service is not None else get_graph_service()
    result = export_graph(active, dest, read_stale=read_stale, on_progress=on_progress)
    counts = result.counts
    logger.info(
        "graph export wrote %s: %d file(s), %d thread-day(s), %d bytes in %.1fs",
        result.bundle.name,
        result.files,
        counts.get("thread_days", 0),
        result.bundle.size_bytes,
        result.wall_clock_s,
    )
    return result
