"""Backup routes: list, export, restore, status (SSE), download (ADR-019).

The ``/corpus`` "Export & restore" panel's whole server surface, mirroring
the ingest routes' shape — a ``202`` start, a status stream that replays the
backlog to late joiners, the structured error envelope:

* ``GET /api/backups/{corpus}`` lists the bundles in the corpus's reserved
  ``varagity-exports/`` folder, newest first, each with its ``bundle.json``
  (read alone through the zip's central directory) and the cheap restore
  verdict — ``restorable``, the ``reason`` when not, and the ingest-setting
  ``drift`` that restoring it would flag stale. A file copied in by hand is
  listed too; an in-flight ``.partial`` never is.
* ``POST /api/backups/{corpus}/export`` starts an export run. Everything
  that can refuse it is checked first, so a refusal is a clean status, never
  a failed run: postgres unreachable (``503``), nothing ingested — or, for
  the graph, nothing ever built (``422 nothing_to_export``) — the folder
  unwritable (``500``), another operation holding the corpus (``409
  corpus_busy``: an ingest or build, another export, a restore).
* ``POST /api/backups/{corpus}/restore`` starts a restore run, with the
  same up-front posture for everything a cheap look can see: the stores it
  writes unreachable (``503``), no such bundle (``404 bundle_not_found``),
  the corpus busy here (``409``), the bundle unsound — or already found
  damaged by an earlier restore — (``422 bundle_invalid``) or not this
  stack's to restore (``422 bundle_incompatible`` — another embedding
  space, or for the graph another engine or engine version), and for RAG an
  ingest in another process holding the corpus lock (``409 corpus_busy``).
  The sha256 of every member is checked by the run itself, as it loads the
  bundle — one read, with progress, and before anything takes effect.
* ``GET /api/backups/{corpus}/status`` streams the run's ``status`` /
  ``progress`` / ``log`` events.
* ``GET /api/backups/{corpus}/files/{name}`` returns a bundle as an
  attachment — a plain navigation, so the browser downloads it with no
  CORS involved — resolving ``name`` inside the folder or answering
  ``404 bundle_not_found``.

**Why neither the list nor the route hashes.** A sha256 pass reads the
whole bundle: the list is loaded on every panel visit, and a restore reads
the bundle anyway to load it. So both show the cheap verdict, and the
restore run — which verifies every member as it loads it, before its
transaction commits or its staged graph is swapped in — *remembers* a
bundle it refused, by name, size and mtime, in this process: from then on
the list shows that bundle as damaged and the route refuses it, instead of
offering it again, until the file changes.

**Why "save to the folder" rather than "open the folder".** No browser API
can open a folder on the user's machine, and the API runs in a container
that can't reach the desktop; the bundle lands where the ticket wants it
(beside the sources, inside the bind mount that survives ``down -v``), and
the panel shows that path with Copy-path and Download.

``corpus`` is ``rag`` or ``graph`` (``ChatRequest.corpus``'s vocabulary) on
every route. While ``GRAPH_ENABLED=false``, every graph route answers ``403
graph_disabled`` — except the status stream, which (like the graph build's)
keeps reporting, so the tab can still render whatever ran.
"""

import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent

from varagity.api.backup_runner import (
    BackupRunner,
    DamagedBundles,
    current_ingest_settings,
    get_backup_runners,
    get_damaged_bundles,
)
from varagity.api.corpus_gate import CorpusBusy
from varagity.api.deps import corpus_busy_error, get_backup_preflight, get_vector_store
from varagity.api.schemas import (
    BackupRestoreRequest,
    BackupRunOut,
    BundleManifestOut,
    BundleOut,
    ErrorResponse,
)
from varagity.backup.bundle import (
    BundleIncompatible,
    BundleInvalid,
    BundleListing,
    BundleManifest,
    display_path,
    exports_dir,
    inspect_bundle,
    is_bundle_name,
    list_bundles,
)
from varagity.backup.graph import workdir_files
from varagity.backup.rag import ingest_drift
from varagity.config import get_settings
from varagity.graph.service import graph_workdir
from varagity.paths import resolve_contained
from varagity.stores.vector_store import ContextualVectorDB

logger = logging.getLogger(__name__)

router = APIRouter(tags=["backups"])

BackupCorpus = Literal["rag", "graph"]
"""The corpora with backups — ``ChatRequest.corpus``'s vocabulary."""

# The structured 500 code per corpus when its exports folder can't be
# written — the same codes the corpus's upload route uses for its root.
_NOT_WRITABLE_CODES = {"rag": "docs_path_not_writable", "graph": "graph_docs_path_not_writable"}


def enabled_corpus(corpus: BackupCorpus) -> BackupCorpus:
    """Resolve the ``{corpus}`` path parameter, refusing the graph while it is switched off.

    A dependency rather than a line in each route body, so it runs *first*
    — before the routes' store dependencies open a connection — and a
    disabled graph answers ``403`` whatever else is down (ADR-017's degrade
    semantics: a refusal, never a silent no-op).

    Args:
        corpus: The path's corpus.

    Returns:
        The same corpus.

    Raises:
        HTTPException: ``403 graph_disabled`` for ``graph`` while
            ``GRAPH_ENABLED`` is false.
    """
    if corpus == "graph" and not get_settings().GRAPH_ENABLED:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "graph_disabled",
                "message": "the graph subsystem is disabled (GRAPH_ENABLED=false) — "
                "turn it on in settings to back up the graph",
            },
        )
    return corpus


EnabledCorpusDep = Annotated[BackupCorpus, Depends(enabled_corpus)]
RunnersDep = Annotated[dict[str, BackupRunner], Depends(get_backup_runners)]
PreflightDep = Annotated[Callable[[str, str], Awaitable[None]], Depends(get_backup_preflight)]
VectorStoreDep = Annotated[ContextualVectorDB, Depends(get_vector_store)]
DamagedDep = Annotated[DamagedBundles, Depends(get_damaged_bundles)]


def _bundle_out(
    listing: BundleListing,
    corpus: str,
    current: Mapping[str, str],
    damaged: DamagedBundles,
) -> BundleOut:
    """Render one listed bundle, with its verdict, for the wire.

    Args:
        listing: The bundle file, its manifest (if readable) and verdict.
        corpus: The corpus the folder belongs to.
        current: This stack's reingest-affecting settings (the drift
            baseline).
        damaged: The remembered restore failures.

    Returns:
        The ``BundleOut`` entry.
    """
    manifest = listing.manifest
    restorable, reason = listing.verdict.restorable, listing.verdict.reason
    remembered = damaged.reason(corpus, listing)
    if restorable and remembered is not None:
        restorable, reason = False, remembered
    drift = (
        ingest_drift(manifest, current)
        if manifest is not None and manifest.corpus == "rag" == corpus
        else []
    )
    return BundleOut(
        name=listing.name,
        size_bytes=listing.size_bytes,
        modified_at=listing.modified_at,
        manifest=None if manifest is None else BundleManifestOut(**manifest.model_dump()),
        restorable=restorable,
        reason=reason,
        drift=drift,
    )


def _ensure_writable_exports_dir(corpus: BackupCorpus) -> Path:
    """Create the corpus's exports folder and confirm the API can write it.

    Args:
        corpus: The corpus about to be exported.

    Returns:
        The exports folder.

    Raises:
        HTTPException: ``500 docs_path_not_writable`` (``graph_docs_path_not_writable``
            for the graph) when the folder can't be created or written (in
            compose, the ``./docs`` / ``./graph-docs`` bind mount must be
            writable by the api container's user — or a *file* named
            ``varagity-exports`` is squatting on the folder's name).
    """
    directory = exports_dir(corpus)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        writable = os.access(directory, os.W_OK | os.X_OK)
        problem = None if writable else "permission denied"
    except OSError as error:
        problem = str(error)
    if problem is not None:
        logger.error("cannot write the exports folder %s: %s", directory, problem)
        raise HTTPException(
            status_code=500,
            detail={
                "code": _NOT_WRITABLE_CODES[corpus],
                "message": (
                    f"the API cannot write {display_path(corpus)} ({problem}) — in compose, "
                    "the corpus bind mount must be writable by the api container's user "
                    "(see the runbook)"
                ),
            },
        )
    return directory


def _bundle_path(corpus: BackupCorpus, name: str) -> Path:
    """Resolve a bundle name inside the corpus's exports folder.

    The containment backstop every user-supplied name gets: a hidden or
    non-``.zip`` name, a path, or a symlink leading out of the folder is
    simply not a bundle this folder holds.

    Args:
        corpus: The corpus.
        name: The requested bundle file name.

    Returns:
        The bundle's resolved path.

    Raises:
        HTTPException: ``404 bundle_not_found`` for anything that isn't a
            bundle file inside the folder.
    """
    directory = exports_dir(corpus)
    target: Path | None = None
    if is_bundle_name(name) and directory.is_dir():
        target = resolve_contained(directory / name, directory.resolve())
    if target is None or not target.is_file():
        raise HTTPException(
            status_code=404,
            detail={
                "code": "bundle_not_found",
                "message": f"No backup named {name!r} in {display_path(corpus)}",
            },
        )
    return target


@router.get("/api/backups/{corpus}", responses={403: {"model": ErrorResponse}})
def list_backups(corpus: EnabledCorpusDep, damaged: DamagedDep) -> list[BundleOut]:
    """List the backup bundles in a corpus's ``varagity-exports/`` folder.

    A directory read plus one small member per bundle — the manifest is
    found through each zip's central directory, so a folder of large
    bundles lists in milliseconds. Each entry carries the cheap restore
    verdict (no hashing — a restore hashes, and a damage it finds is
    remembered here).

    Args:
        corpus: The corpus whose backups to list.
        damaged: The process's remembered restore failures.

    Returns:
        One entry per bundle file, newest first (empty when the folder does
        not exist yet).

    Raises:
        HTTPException: ``403 graph_disabled`` for the graph while
            ``GRAPH_ENABLED`` is false.
    """
    # Drift is a RAG notion (the graph records no ingest settings).
    current = current_ingest_settings() if corpus == "rag" else {}
    return [
        _bundle_out(listing, corpus, current, damaged)
        for listing in list_bundles(exports_dir(corpus), corpus)
    ]


def _nothing_to_export(corpus: BackupCorpus, store: ContextualVectorDB) -> str | None:
    """Say why a corpus has nothing to back up, if it hasn't.

    Args:
        corpus: The corpus about to be exported.
        store: The per-request vector store (the RAG corpus's row count).

    Returns:
        The refusal's message, or ``None`` when there is something to export
        — RAG: any ``documents`` row; graph: any file the export would copy
        from the engine's workdir (the export's own rule, so the two never
        disagree).
    """
    if corpus == "rag":
        if store.document_count() == 0:
            return "the corpus holds no documents — ingest something before exporting"
        return None
    if not workdir_files(graph_workdir()):
        return "nothing has been built into the graph yet — build it before exporting"
    return None


@router.post(
    "/api/backups/{corpus}/export",
    status_code=202,
    responses={
        403: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def start_export(
    corpus: EnabledCorpusDep,
    runners: RunnersDep,
    preflight: PreflightDep,
    store: VectorStoreDep,
) -> BackupRunOut:
    """Start a background export and return its handle.

    The bundle is written into ``<corpus root>/varagity-exports/`` as
    ``varagity-<corpus>-<UTC yyyymmdd-HHMM>.zip`` (a same-minute repeat gets
    a ``-2`` suffix — never a replace): for RAG, from one consistent
    snapshot of the corpus; for the graph, the engine's workdir copied under
    the graph's write lock, while graph chat keeps answering.

    Args:
        corpus: The corpus to export.
        runners: The process-wide backup runners.
        preflight: The awaitable reachability check (structured 503).
        store: The per-request vector store (the RAG nothing-to-export
            check).

    Returns:
        The new run's snapshot (state ``running``).

    Raises:
        HTTPException: ``403 graph_disabled`` for the graph while
            ``GRAPH_ENABLED`` is false; ``503 postgres_unreachable`` when
            the database is down; ``422 nothing_to_export`` when nothing is
            ingested (or, for the graph, nothing has been built); ``500
            docs_path_not_writable`` / ``graph_docs_path_not_writable`` when
            the exports folder can't be written; ``409 corpus_busy`` while
            any operation — another export, an ingest or build, a restore —
            holds the corpus (the message names it).
    """
    await preflight(corpus, "export")
    refusal = await run_in_threadpool(_nothing_to_export, corpus, store)
    if refusal is not None:
        raise HTTPException(
            status_code=422,
            detail={"code": "nothing_to_export", "message": refusal},
        )
    await run_in_threadpool(_ensure_writable_exports_dir, corpus)
    try:
        return runners[corpus].start("export")
    except CorpusBusy as error:
        raise corpus_busy_error(error) from error


def _restorable_manifest(path: Path, corpus: BackupCorpus) -> BundleManifest:
    """The cheap verdict a restore is started on — the list's own check.

    The zip's central directory and ``bundle.json`` alone, never a member's
    bytes: the run verifies every member's sha256 as it loads it.

    Args:
        path: The bundle file.
        corpus: The corpus it would restore.

    Returns:
        The bundle's manifest.

    Raises:
        BundleInvalid: If the bundle is unsound.
        BundleIncompatible: If this stack must not restore it.
    """
    return inspect_bundle(path, corpus).manifest_or_raise()


@router.post(
    "/api/backups/{corpus}/restore",
    status_code=202,
    responses={
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def start_restore(
    corpus: EnabledCorpusDep,
    payload: BackupRestoreRequest,
    runners: RunnersDep,
    preflight: PreflightDep,
    store: VectorStoreDep,
    damaged: DamagedDep,
) -> BackupRunOut:
    """Start a background restore of one bundle and return its handle.

    **Restore replaces**, with no model call. For RAG, the corpus's rows in
    both stores become the bundle's, re-rooted onto this stack's
    ``DOCS_PATH`` — nothing parsed, chunked, contextualized or embedded —
    and the BM25 index is rebuilt from the restored rows. For the graph, the
    engine's working directory becomes the bundle's — extracted beside the
    live one and swapped in, the previous graph put back if the restored one
    won't open — and nothing is re-extracted. Everything a cheap look can
    see refuses the bundle here, before the ``202``: the list's verdict, and
    a damage an earlier restore of this very file found. The sha256 of every
    member is the run's job — it reads the bundle once, to load it, and
    nothing takes effect until every member checked out — so a bundle
    damaged inside an intact zip fails its first run (the corpus untouched)
    and is refused here from then on.

    Args:
        corpus: The corpus to restore.
        payload: The bundle to restore, by name.
        runners: The process-wide backup runners.
        preflight: The awaitable reachability check (structured 503).
        store: The per-request vector store (the RAG corpus-lock check).
        damaged: The process's remembered restore failures (a bundle an
            earlier restore found damaged is refused here, before the run).

    Returns:
        The new run's snapshot (state ``running``).

    Raises:
        HTTPException: ``403 graph_disabled`` for the graph while
            ``GRAPH_ENABLED`` is false; ``503 postgres_unreachable`` /
            ``es_unreachable`` when a store the restore writes is down;
            ``404 bundle_not_found`` for a name that is not a bundle in the
            folder; ``409 corpus_busy`` while another operation holds the
            corpus, or (RAG) an ingest in another process holds the corpus
            lock; ``422 bundle_invalid`` for an unsound bundle, or one an
            earlier restore found damaged; ``422 bundle_incompatible`` for
            one from another embedding model
            or dimension, corpus, or format version — or, for the graph,
            another engine or engine major.minor version.
    """
    await preflight(corpus, "restore")
    path = _bundle_path(corpus, payload.name)
    runner = runners[corpus]
    try:
        # First: a busy corpus shouldn't cost even the manifest read.
        runner.ensure_free()
    except CorpusBusy as error:
        raise corpus_busy_error(error) from error
    try:
        await run_in_threadpool(_restorable_manifest, path, corpus)
    except BundleIncompatible as error:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "bundle_incompatible",
                "message": f"cannot restore {payload.name}: {error}",
            },
        ) from error
    except BundleInvalid as error:
        raise _bundle_invalid(payload.name, str(error)) from error
    remembered = await run_in_threadpool(damaged.reason_for, corpus, payload.name, path)
    if remembered is not None:
        raise _bundle_invalid(payload.name, remembered)
    # The cross-process half of the gate is RAG's alone: the graph's only
    # writer is this process, so its gate already says everything.
    if corpus == "rag" and not await run_in_threadpool(store.corpus_lock_free):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "corpus_busy",
                "message": (
                    f"the {corpus} corpus is busy — an ingest in another process holds the corpus "
                    "lock (a CLI `main.py ingest` or `chat`); try again when it finishes"
                ),
            },
        )
    try:
        return runner.start("restore", payload.name)
    except CorpusBusy as error:
        raise corpus_busy_error(error) from error


def _bundle_invalid(name: str, reason: str) -> HTTPException:
    """Build the structured ``422 bundle_invalid`` for a bundle that can't restore.

    Args:
        name: The bundle's file name.
        reason: Why (the verdict's, or a damage an earlier restore found).

    Returns:
        The ``422 bundle_invalid`` to raise.
    """
    return HTTPException(
        status_code=422,
        detail={"code": "bundle_invalid", "message": f"cannot restore {name}: {reason}"},
    )


@router.get("/api/backups/{corpus}/status", response_class=EventSourceResponse)
async def backup_status(
    corpus: BackupCorpus, runners: RunnersDep
) -> AsyncIterator[ServerSentEvent]:
    """Stream the corpus's current (or last) backup run as SSE.

    Frames: ``status`` (snapshot; also terminal, with the summary — and an
    export's saved path), ``progress`` (a RAG export's ``snapshot`` →
    ``documents`` → ``chunks`` → ``finalize``; a graph export's ``snapshot``
    → ``pack``, in bytes → ``finalize``; a RAG restore's ``validate`` →
    ``load`` → ``index`` → ``bm25``; a graph restore's ``validate`` →
    ``extract``, in bytes → ``swap`` → ``open``), ``log`` (relayed
    ``varagity.backup`` lines).
    With no run in this API process the stream is a single idle ``status``
    frame; after a terminal run it replays that run and closes. Never
    refused for a disabled graph: a tab must still render what ran.

    Args:
        corpus: The corpus whose backup run to follow.
        runners: The process-wide backup runners.

    Yields:
        The framed events, oldest first, then live until terminal.
    """
    async for event, payload in runners[corpus].subscribe():
        yield ServerSentEvent(event=event, data=payload)


@router.get(
    "/api/backups/{corpus}/files/{name}",
    response_class=FileResponse,
    responses={
        200: {
            "description": "The bundle, as a `Content-Disposition: attachment` download.",
            "content": {"application/zip": {"schema": {"type": "string", "format": "binary"}}},
        },
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
def download_backup(corpus: EnabledCorpusDep, name: str) -> FileResponse:
    """Return one bundle as a file download.

    A plain ``GET`` answering with ``Content-Disposition: attachment``, so
    the panel's Download is an ordinary link — the browser saves the file
    (and offers its own "Show in folder") without a cross-origin fetch.

    Args:
        corpus: The corpus the bundle belongs to.
        name: The bundle's file name in the exports folder.

    Returns:
        The bundle's bytes as ``application/zip``.

    Raises:
        HTTPException: ``403 graph_disabled`` for the graph while
            ``GRAPH_ENABLED`` is false; ``404 bundle_not_found`` when no such
            bundle lives in the folder — which covers a name trying to
            escape it.
    """
    path = _bundle_path(corpus, name)
    return FileResponse(path, media_type="application/zip", filename=name)
