"""Prefect backup flows: export and restore, for both corpora (ADR-019).

Thin ``@flow`` shells over :func:`varagity.backup.rag.export_rag`,
:func:`varagity.backup.rag.restore_rag`,
:func:`varagity.backup.graph.export_graph` and
:func:`varagity.backup.graph.restore_graph`, for the reason the graph build
is one: the work belongs to the backup package (one corpus snapshot and one
all-or-nothing bundle write; one replacing transaction and one index
rebuild; one copy of the graph's workdir under its write lock; one extract
and one swap of it), while Prefect owns the *record* that a backup ran, how
long it took, and whether it failed. One flow run per operation — not a task
per member or per stage: an export is a single consistent read and a restore
a single transaction (or a single swap), and slicing either into task runs
would invent boundaries they do not have. Per-stage progress travels as the
API runner's ``progress`` events instead (tracking and streaming compose, as
they do for ingest and the graph build).

No Prefect-level retries: a failed export leaves no bundle behind (the
partial is removed) and a failed restore leaves the previous corpus (its
transaction rolled back, or its previous workdir put back) — or, failing in
the BM25 rebuild after the commit, says so — so the honest response is to
surface the failure and let the user press the button again, not to repeat
the work silently. Parameter validation is off because the stores (and the
graph service) are live handles, not serializable payloads.
"""

from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

from prefect import flow
from prefect.logging import get_run_logger

from varagity.backup.bundle import BundleManifest
from varagity.backup.graph import (
    GraphExportProgressCallback,
    GraphExportReport,
    GraphRestore,
    GraphRestoreProgressCallback,
    export_graph,
    restore_graph,
)
from varagity.backup.rag import (
    BM25RebuildFailed,
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
    stack's ``DOCS_PATH``) and rebuilds the BM25 index from them, then
    writes the corpus-stale flag the bundle implies: stale if it was stale
    at export, or if its rows were made under other reingest-affecting
    settings than this stack's. The flag follows the rows: it is written
    once they committed — after a complete restore, and also when only the
    BM25 rebuild failed (the rows are the bundle's by then, so the previous
    corpus's flag would misdescribe them) — and never after a restore that
    rolled back.

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

    def write_stale(manifest: BundleManifest) -> bool:
        stale = derive_corpus_stale(manifest, ingest_settings())
        if settings_store is not None:
            settings_store.set_corpus_stale(stale)
        else:
            with AppSettingsStore() as owned_settings:
                owned_settings.set_corpus_stale(stale)
        return stale

    try:
        with ExitStack() as stack:
            active_store = store if store is not None else stack.enter_context(ContextualVectorDB())
            active_bm25 = bm25 if bm25 is not None else stack.enter_context(ElasticsearchBM25())
            result = restore_rag(
                bundle, active_store, active_bm25, docs_root, on_progress=on_progress
            )
    except BM25RebuildFailed as error:
        # The rows committed before the rebuild failed: they are the bundle's
        # now, so the flag must describe them too — then the run still fails.
        if error.manifest is not None:
            try:
                stale = write_stale(error.manifest)
            except Exception:  # the rebuild's failure stays the run's answer
                logger.exception("rag restore could not write the corpus-stale flag")
            else:
                logger.warning(
                    "rag restore committed the rows of %s but not the BM25 index%s",
                    bundle.name,
                    " — the corpus is flagged stale" if stale else "",
                )
        raise
    stale = write_stale(result.manifest)
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


@flow(name="graph-restore", validate_parameters=False)
def graph_restore_flow(
    bundle: Path,
    *,
    run_id: str,
    on_progress: GraphRestoreProgressCallback | None = None,
    service: GraphService | None = None,
    settings_store: AppSettingsStore | None = None,
) -> GraphRestore:
    """Restore the message graph from a bundle as a tracked flow run.

    Extracts the bundle's workdir beside the live one and swaps it in under
    the graph service's write and open locks — the previous graph comes back
    if the restored one won't open — then, and only once all of that
    succeeded, writes the graph-stale flag the bundle recorded. Needs no
    model service: only postgres, for the flag.

    Args:
        bundle: The bundle file (inside ``<GRAPH_DOCS_PATH>/varagity-exports``).
        run_id: The restore run's id (names the staging and previous folders
            beside the workdir).
        on_progress: Called with ``(stage, current, total)`` per stage
            (``validate`` → ``extract``, counting bytes → ``swap`` →
            ``open``).
        service: The graph service whose workdir to replace; the process's
            own (the graph's single writer) when omitted.
        settings_store: Where the stale flag is written; a connection of its
            own is opened (and closed) when omitted.

    Returns:
        What was restored, with the stale flag it left.
    """
    logger = get_run_logger()
    active = service if service is not None else get_graph_service()
    result = restore_graph(active, bundle, run_id=run_id, on_progress=on_progress)
    # The bundle's own flag: the graph is exactly what was exported, so it is
    # stale exactly when it was then (a source deleted before the export).
    stale = result.manifest.stale
    if settings_store is not None:
        settings_store.set_graph_stale(stale)
    else:
        with AppSettingsStore() as owned_settings:
            owned_settings.set_graph_stale(stale)
    logger.info(
        "graph restore loaded %s: %d file(s), %d thread-day(s) in %.1fs%s",
        result.bundle_name,
        result.files,
        result.counts.get("thread_days", 0),
        result.wall_clock_s,
        " — the graph is flagged stale" if stale else "",
    )
    return replace(result, stale=stale)
