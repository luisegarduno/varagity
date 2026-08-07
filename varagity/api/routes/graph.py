"""Graph corpus routes: upload, list, delete, build, status (spec_graphrag §5.2).

The Graph RAG tab's whole server surface, and the deliberate mirror of the
chunk-RAG corpus routes (:mod:`varagity.api.routes.documents` +
:mod:`varagity.api.routes.ingest`) with the differences the subject matter
forces:

* **Uploads are sniffed, not allow-listed.** ``ALLOWED_EXTENSIONS`` governs
  the chunk-RAG corpus and says nothing here (``.db`` isn't even in it);
  what decides is whether a registered
  :class:`~varagity.graph.sources.base.MessageSource` claims the *stored
  bytes* — the ``imessage`` probe checks the SQLite magic header and the
  presence of a ``message`` table (behind a cheap ``.db`` suffix gate of its
  own). A file that lands and then fails the sniff is removed again and
  reported as ``unsupported_graph_source``. The cap is its own setting
  (``GRAPH_UPLOAD_MAX_MB``, default 4 GB): a decade of iMessage history is
  one enormous file, where the corpus cap is document-sized.
  **WAL sidecars are first-class**: a live ``chat.db`` keeps recent messages
  in ``chat.db-wal``, so copying all three files (``.db``, ``-wal``,
  ``-shm``) is the documented rule — the sidecars are stored beside their
  database and never sniffed on their own.
* **The listing is a directory scan**, not a table read. The graph's own
  record of what it indexed lives in the engine's workdir manifest; what
  this tab has to show is what is on disk, including files the user dropped
  in that no source claims. Parse summaries come from the last build's scan
  (parsing a multi-gigabyte database is a build's job, not a GET's).
* **Deleting a source file flags the graph stale.** The graph still holds
  that file's messages — only a ``reingest`` rebuild retracts them — so the
  flag is set and the tab says so (stage-2 decision #16).
* **Builds are API-only** (stage-2 decision #4): the engine's storages are
  single-writer per working directory, and this process owns them. There is
  deliberately no CLI graph build.

``GRAPH_ENABLED=false`` turns the two *mutating* entry points — upload and
build — into a structured ``403 graph_disabled`` (ADR-017's degrade
semantics: never a silent no-op). Listing, deleting, and the status poll
keep working, because a disabled graph still has a corpus directory to
manage and a tab that must render honestly.
"""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.sse import EventSourceResponse, ServerSentEvent

from varagity.api.deps import get_app_settings_store, get_graph_build_preflight
from varagity.api.graph_runner import (
    GraphBuildAlreadyRunning,
    GraphBuildRunner,
    get_graph_build_runner,
)
from varagity.api.routes.documents import _store_upload
from varagity.api.schemas import (
    ErrorResponse,
    GraphBuildRequest,
    GraphBuildRunOut,
    GraphDocumentDeleteResponse,
    GraphDocumentOut,
    GraphStatusOut,
    GraphUploadResponse,
    UploadedFileOut,
)
from varagity.config import get_settings
from varagity.graph.manifest import load_summary
from varagity.graph.service import GraphService, GraphUnavailable, get_graph_service
from varagity.graph.sources.base import find_message_source
from varagity.paths import resolve_contained
from varagity.stores.app_settings_store import AppSettingsStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["graph"])

RunnerDep = Annotated[GraphBuildRunner, Depends(get_graph_build_runner)]
ServiceDep = Annotated[GraphService, Depends(get_graph_service)]
SettingsStoreDep = Annotated[AppSettingsStore, Depends(get_app_settings_store)]

# Sidecars a live SQLite database keeps beside itself. They carry the most
# recent messages, so they must be uploadable — but they are not databases
# on their own and must never be sniffed as one.
_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")


def _require_enabled() -> None:
    """Refuse a mutating graph operation while the kill switch is off.

    Raises:
        HTTPException: ``403 graph_disabled`` when ``GRAPH_ENABLED`` is
            false. A structured refusal, not a silent no-op: the caller
            asked to change the graph and nothing happened (ADR-017).
    """
    if not get_settings().GRAPH_ENABLED:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "graph_disabled",
                "message": "the graph subsystem is disabled (GRAPH_ENABLED=false) — "
                "turn it on in settings to upload or build",
            },
        )


def _is_sidecar(name: str) -> bool:
    """Report whether a file name is a SQLite sidecar rather than a database.

    Args:
        name: The stored file's name.

    Returns:
        ``True`` for ``…-wal`` / ``-shm`` / ``-journal`` companions.
    """
    return any(name.endswith(suffix) for suffix in _SIDECAR_SUFFIXES)


def _sniff_stored(result: UploadedFileOut, graph_root: Path) -> UploadedFileOut:
    """Confirm a just-stored upload really is a graph source, or remove it.

    Args:
        result: The write outcome from the shared upload path.
        graph_root: The resolved ``GRAPH_DOCS_PATH``.

    Returns:
        The outcome unchanged when the file is a sidecar or a registered
        source claims it; otherwise a rejection carrying
        ``unsupported_graph_source``, with the stored file removed again
        (a corpus directory must not accumulate files nothing can read).
    """
    if not result.stored or _is_sidecar(result.file_name):
        return result
    target = graph_root / result.file_name
    if find_message_source(target) is not None:
        return result
    logger.info("removing %s — no registered message source claims it", target)
    target.unlink(missing_ok=True)
    return UploadedFileOut(
        file_name=result.file_name,
        size_bytes=0,
        stored=False,
        reason="unsupported_graph_source",
    )


@router.post("/api/graph/documents", status_code=201, responses={403: {"model": ErrorResponse}})
def upload_graph_documents(files: list[UploadFile]) -> GraphUploadResponse:
    """Upload message-source file(s) into ``GRAPH_DOCS_PATH`` (no auto-build).

    Each file is written independently and then **sniffed**: a file no
    registered message source claims is deleted again and reported as
    ``unsupported_graph_source``, so a mixed drop partially succeeds and the
    corpus directory never fills with unreadable files. The chunk-RAG
    ``ALLOWED_EXTENSIONS`` list does not apply — contents decide.

    Copy the whole SQLite set — ``chat.db`` **and** its ``-wal``/``-shm``
    sidecars: a live database keeps its most recent messages in the
    write-ahead log, and uploading the ``.db`` alone silently loses them.
    Sidecars are stored beside their database without being sniffed.

    Args:
        files: The multipart file parts.

    Returns:
        Per-file outcomes, in upload order.

    Raises:
        HTTPException: ``403 graph_disabled`` when the kill switch is off;
            ``422 too_many_files`` when the batch busts ``UPLOAD_MAX_FILES``;
            ``422 no_file_stored`` when every file was rejected on its own
            merits; ``500 graph_docs_path_not_writable`` when nothing landed
            because the server couldn't write ``GRAPH_DOCS_PATH`` (in
            compose, the ``./graph-docs`` bind mount must be writable by the
            api container's user).
    """
    _require_enabled()
    settings = get_settings()
    if len(files) > settings.UPLOAD_MAX_FILES:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "too_many_files",
                "message": f"{len(files)} files exceed UPLOAD_MAX_FILES "
                f"({settings.UPLOAD_MAX_FILES}) — split the upload",
            },
        )
    graph_root = Path(settings.GRAPH_DOCS_PATH)
    not_writable = HTTPException(
        status_code=500,
        detail={
            "code": "graph_docs_path_not_writable",
            "message": (
                f"the API cannot write GRAPH_DOCS_PATH ({graph_root}) — in compose, the "
                "./graph-docs bind mount must be writable by the api container's user "
                "(see the runbook)"
            ),
        },
    )
    try:
        graph_root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        logger.error("cannot create GRAPH_DOCS_PATH %s: %s", graph_root, error)
        raise not_writable from error
    # The corpus routes' per-file streaming write, with the graph's own cap
    # and no extension gate: format is decided by the sniff below, not by a
    # name (a chat.db copy is routinely renamed).
    max_bytes = settings.GRAPH_UPLOAD_MAX_MB * 1024 * 1024
    results = [
        _sniff_stored(_store_upload(upload, graph_root, max_bytes), graph_root) for upload in files
    ]
    if not any(result.stored for result in results):
        if any(result.reason == "write_failed" for result in results):
            raise not_writable
        raise HTTPException(
            status_code=422,
            detail={
                "code": "no_file_stored",
                "message": "; ".join(f"{r.file_name}: {r.reason}" for r in results)
                or "empty upload",
            },
        )
    return GraphUploadResponse(files=results)


def _as_utc(epoch_seconds: float) -> datetime:
    """Render a filesystem timestamp as an aware UTC datetime.

    Args:
        epoch_seconds: The stat result's epoch seconds.

    Returns:
        The timestamp in UTC (a naive local one would be a lie on the wire).
    """
    return datetime.fromtimestamp(epoch_seconds, tz=UTC)


@router.get("/api/graph/documents")
def list_graph_documents(runner: RunnerDep) -> list[GraphDocumentOut]:
    """List every file in the graph corpus directory.

    A directory scan, not a table read: the graph's own record of what it
    indexed lives in the engine's workdir, and what this tab must show is
    what has been dropped in — sidecars and contacts files included. Files
    the last build parsed also carry their ``doc_id`` and a parse summary;
    files no build has scanned carry ``None`` for both, because parsing a
    multi-gigabyte database is a build's job.

    Args:
        runner: The process-wide graph build runner (holds the last scan).

    Returns:
        One entry per file, in path order.
    """
    graph_root = Path(get_settings().GRAPH_DOCS_PATH)
    scanned = runner.scanned_documents()
    entries: list[GraphDocumentOut] = []
    if not graph_root.is_dir():
        return entries
    for path in sorted(graph_root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        relative = path.relative_to(graph_root).as_posix()
        stat = path.stat()
        known = scanned.get(relative)
        entries.append(
            GraphDocumentOut(
                name=path.name,
                relative_path=relative,
                size_bytes=stat.st_size,
                modified_at=_as_utc(stat.st_mtime),
                doc_id=known.doc_id if known is not None else None,
                parse=known.parse if known is not None else None,
            )
        )
    return entries


def _prune_empty_parents(directory: Path, graph_root: Path) -> None:
    """Remove now-empty folders a deleted source file leaves behind.

    Best-effort by design, exactly as the corpus route's twin is: ``rmdir``
    refusing a non-empty directory is the stop condition, not an error.

    Args:
        directory: The deleted file's (resolved) parent directory.
        graph_root: The resolved ``GRAPH_DOCS_PATH`` — the exclusive upper
            bound.
    """
    current = directory
    while current != graph_root and current.is_relative_to(graph_root):
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


@router.delete(
    "/api/graph/documents/{name:path}",
    responses={404: {"model": ErrorResponse}},
)
def delete_graph_document(name: str, store: SettingsStoreDep) -> GraphDocumentDeleteResponse:
    """Remove one file from the graph corpus directory and flag the graph stale.

    Deleting the *source* does not retract what the graph already extracted
    from it — that costs a rebuild — so this sets the graph-stale flag and
    the tab surfaces it, rather than letting the graph and its corpus
    diverge silently (stage-2 decision #16).

    Args:
        name: The file's path relative to ``GRAPH_DOCS_PATH``.
        store: The per-request app-settings store (the stale flag).

    Returns:
        What was removed and whether the graph is now flagged stale.

    Raises:
        HTTPException: ``404 graph_document_not_found`` when no such file
            lives inside ``GRAPH_DOCS_PATH`` — which covers a path trying to
            escape it, since an escaping path is not a file this route has.
    """
    graph_root = Path(get_settings().GRAPH_DOCS_PATH).resolve()
    target = resolve_contained(graph_root / name, graph_root)
    if target is None or not target.is_file():
        raise HTTPException(
            status_code=404,
            detail={
                "code": "graph_document_not_found",
                "message": f"No graph corpus file at {name!r}",
            },
        )
    relative = target.relative_to(graph_root).as_posix()
    target.unlink()
    _prune_empty_parents(target.parent, graph_root)
    store.set_graph_stale(True)
    logger.info("deleted graph source %s — graph flagged stale until the next rebuild", relative)
    return GraphDocumentDeleteResponse(relative_path=relative, graph_stale=True)


@router.post(
    "/api/graph/build",
    status_code=202,
    responses={
        403: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def start_graph_build(
    payload: GraphBuildRequest,
    runner: RunnerDep,
    preflight: Annotated[Callable[[], Awaitable[None]], Depends(get_graph_build_preflight)],
) -> GraphBuildRunOut:
    """Trigger a background graph build and return its handle.

    **This is also how a killed build resumes.** The engine keeps document
    statuses on disk and re-selects every in-flight or failed document, so
    pressing Build again after a crash finishes the work instead of
    repeating it — the run's summary reports what it inherited as
    ``resumed``. That is why ``reingest`` is a separate, explicit flag: it
    is the one path that throws the working directory away.

    Args:
        payload: ``reingest`` (wipe and re-index — the only thing that
            clears the graph-stale flag), plus the ``message_limit`` /
            ``since`` bounds that make a minutes-long spot check possible on
            an archive whose full build runs for days.
        runner: The process-wide graph build runner.
        preflight: The awaitable reachability check (structured 503).

    Returns:
        The new run's snapshot (state ``running``).

    Raises:
        HTTPException: ``403 graph_disabled`` when the kill switch is off;
            ``409 graph_build_running`` while a build is in flight; ``503
            <service>_unreachable`` when a required model service is down.
    """
    _require_enabled()
    await preflight()
    try:
        return runner.start(
            reingest=payload.reingest,
            message_limit=payload.message_limit,
            since=payload.since,
        )
    except GraphBuildAlreadyRunning as error:
        raise HTTPException(
            status_code=409,
            detail={"code": "graph_build_running", "message": str(error)},
        ) from error


@router.get("/api/graph/build/status", response_class=EventSourceResponse)
async def graph_build_status(runner: RunnerDep) -> AsyncIterator[ServerSentEvent]:
    """Stream the current (or last) graph build's progress as SSE.

    Frames: ``status`` (snapshot; also terminal, with the summary),
    ``progress`` (``scan`` → ``parse`` per file → ``bound``/``reset`` →
    ``index`` → sampled ``process`` ticks), ``log`` (relayed
    ``varagity.graph`` lines). The stream replays from the start, so a
    browser opened six hours into a backfill renders the same picture one
    opened at the start did.

    Args:
        runner: The process-wide graph build runner.

    Yields:
        The framed events, oldest first, then live until terminal.
    """
    async for event, payload in runner.subscribe():
        yield ServerSentEvent(event=event, data=payload)


@router.get("/api/graph/status")
def graph_status(runner: RunnerDep, service: ServiceDep, store: SettingsStoreDep) -> GraphStatusOut:
    """Report the graph's size, staleness, and build state (the tab's poll).

    Reads the workdir's summary sidecar rather than the graph itself, so a
    poll never walks a multi-megabyte graphml (stage-2 decision #18), and
    skips the engine entirely when nothing has been built here — an
    unbuilt graph must not make a status poll open a session and pay an
    engine's startup.

    Args:
        runner: The process-wide graph build runner (build state).
        service: The process-wide graph service (document statuses).
        store: The per-request app-settings store (the stale flag).

    Returns:
        The status snapshot; every size field is ``None``/empty when
        nothing has been indexed, never a zero that would read like an
        empty graph.
    """
    settings = get_settings()
    status = GraphStatusOut(
        enabled=settings.GRAPH_ENABLED,
        stale=store.is_graph_stale(),
        building=runner.building,
        last_build=runner.snapshot(),
    )
    if not service.workdir.is_dir():
        return status
    summary = load_summary(service.workdir)
    if summary is not None:
        status.entities = summary.entities
        status.relations = summary.relations
        status.message_guids = summary.message_guids
    try:
        status.documents = service.document_statuses()
    except GraphUnavailable as error:
        logger.warning("graph status: the session would not open (%s)", error)
    return status
