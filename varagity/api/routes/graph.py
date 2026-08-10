"""Graph routes: corpus upload/list/delete, build, status, export (§5.2, §4.4).

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
* **Reading the graph never builds one.** The export and entity-detail
  routes — the graph view's data source (spec_graphrag §4.4) — skip the
  engine entirely when the working directory does not exist, because
  opening a session creates the storages and pays the tokenizer's startup.
  A page load must not turn "nothing indexed yet" into an initialized
  workdir.

``GRAPH_ENABLED=false`` turns every entry point that *acts on* the graph —
upload, build, and the two read routes the view draws from — into a
structured ``403 graph_disabled`` (ADR-017's degrade semantics: never a
silent no-op). Listing, deleting, and the status poll keep working, because
a disabled graph still has a corpus directory to manage and a tab that must
render honestly.
"""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
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
    GraphEntityDetailOut,
    GraphExportEdgeOut,
    GraphExportNodeOut,
    GraphExportOut,
    GraphStatusOut,
    GraphTranscriptRefOut,
    GraphUploadResponse,
    UploadedFileOut,
)
from varagity.config import get_settings
from varagity.graph.manifest import load_manifest, load_summary
from varagity.graph.records import GraphExport, GraphExportEdge, GraphExportNode
from varagity.graph.service import GraphService, GraphUnavailable, get_graph_service
from varagity.graph.sources.base import find_message_source
from varagity.paths import resolve_contained
from varagity.stores.app_settings_store import AppSettingsStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["graph"])

# The graph view's ceiling (stage-2 decision #19; raised 2026-08-10). The
# decision sized it from the eval corpus (347 entities per 10k messages);
# the real archive's graph outgrew the old 1 000-node default slice, so the
# owner raised the drawn slice to 5 000. Still finite on purpose: a browser
# asked to lay out an unbounded WebGL graph is a hung tab, and the honest
# answer to "more than this" is `truncated`, not a slow page. The engine
# clamps every slice to its import-time `MAX_GRAPH_NODES`, so the adapter
# pins that at least this high (a regression test holds the two together).
MAX_EXPORT_NODES = 5000

# Whole-graph selector (the engine's own convention, mirrored on the wire).
_WHOLE_GRAPH = "*"

# Hops a labelled export may walk. The engine's neighbourhood expansion is
# exponential in depth, and the view only ever draws one hop.
_MAX_EXPORT_DEPTH = 5

RunnerDep = Annotated[GraphBuildRunner, Depends(get_graph_build_runner)]
ServiceDep = Annotated[GraphService, Depends(get_graph_service)]
SettingsStoreDep = Annotated[AppSettingsStore, Depends(get_app_settings_store)]

# Sidecars a live SQLite database keeps beside itself. They carry the most
# recent messages, so they must be uploadable — but they are not databases
# on their own and must never be sniffed as one.
_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")


def _require_enabled() -> None:
    """Refuse a graph operation the kill switch must not serve.

    Everything that *acts on* the graph — uploading a source, building, and
    reading the graph itself for the view — is refused; the honest reporting
    surfaces (the corpus listing, the delete, the status poll) keep working,
    because a disabled graph still has a corpus directory to manage and a
    tab that must explain itself.

    Raises:
        HTTPException: ``403 graph_disabled`` when ``GRAPH_ENABLED`` is
            false. A structured refusal, not a silent no-op: the caller
            asked for the graph and nothing happened (ADR-017).
    """
    if not get_settings().GRAPH_ENABLED:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "graph_disabled",
                "message": "the graph subsystem is disabled (GRAPH_ENABLED=false) — "
                "turn it on in settings to upload, build, or read the graph",
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


def _read_graph(
    service: GraphService, label: str, *, max_depth: int, max_nodes: int
) -> GraphExport:
    """Read one slice of the graph, refusing to build a workdir to do it.

    An unbuilt graph answers empty *without opening the engine*: a session
    open creates the storages and pays the tokenizer's startup, which a page
    load must never do to discover there is nothing to draw (the status
    route's discipline — stage-2 decision #18).

    Args:
        service: The process-wide graph service.
        label: Entity name to centre on, or ``"*"`` for the whole graph.
        max_depth: Hops to walk out from ``label``.
        max_nodes: Node cap.

    Returns:
        The slice; empty when nothing has been indexed here yet.

    Raises:
        HTTPException: ``503 graph_unavailable`` when the session will not
            open. Deliberately *not* an empty export: "nothing built yet"
            and "the engine is broken" must not render as the same picture.
    """
    if not service.workdir.is_dir():
        return GraphExport()
    try:
        return service.export(label, max_depth=max_depth, max_nodes=max_nodes)
    except GraphUnavailable as error:
        logger.warning("graph export: the session would not open (%s)", error)
        raise HTTPException(
            status_code=503,
            detail={"code": "graph_unavailable", "message": str(error)},
        ) from error


def _node_out(node: GraphExportNode) -> GraphExportNodeOut:
    """Project one export node onto the wire.

    Args:
        node: The engine-independent record.

    Returns:
        The wire model — deliberately without ``doc_keys``, which a
        whole-graph export would carry by the dozen per node for a picture
        that draws none of them (the entity-detail route resolves them into
        :class:`GraphTranscriptRefOut` instead).
    """
    return GraphExportNodeOut(
        id=node.id,
        entity_type=node.entity_type,
        description=node.description,
        degree=node.degree,
    )


def _edge_out(edge: GraphExportEdge) -> GraphExportEdgeOut:
    """Project one export edge onto the wire.

    Args:
        edge: The engine-independent record.

    Returns:
        The wire model.
    """
    return GraphExportEdgeOut(
        id=edge.id,
        source=edge.source,
        target=edge.target,
        label=edge.label,
        description=edge.description,
    )


@router.get("/api/graph/export", responses={403: {"model": ErrorResponse}})
def export_graph(
    service: ServiceDep,
    label: str = _WHOLE_GRAPH,
    max_depth: Annotated[int, Query(ge=1, le=_MAX_EXPORT_DEPTH)] = 3,
    max_nodes: Annotated[int, Query(ge=1, le=MAX_EXPORT_NODES)] = MAX_EXPORT_NODES,
) -> GraphExportOut:
    """Read a drawable slice of the graph (spec_graphrag §4.4).

    The graph view's whole data source. The default is the *whole* graph,
    degree-ordered so the cap keeps the most connected entities, and
    ``truncated`` says when the cap bit — the view surfaces that rather than
    implying it drew everything.

    Args:
        service: The process-wide graph service.
        label: Entity name to centre the slice on; ``"*"`` (the default)
            takes the whole graph.
        max_depth: Hops to walk out from ``label`` (ignored for ``"*"``).
        max_nodes: Node cap, defaulting to its own ceiling
            :data:`MAX_EXPORT_NODES` (the view draws the fullest slice the
            contract allows). Above it the request is a ``422`` rather than
            a silent clamp: a caller asking for more than the view can
            render should hear so.

    Returns:
        The slice; empty (not an error) when nothing has been indexed yet.

    Raises:
        HTTPException: ``403 graph_disabled`` when the kill switch is off;
            ``503 graph_unavailable`` when the engine will not open.
    """
    _require_enabled()
    export = _read_graph(service, label, max_depth=max_depth, max_nodes=max_nodes)
    return GraphExportOut(
        nodes=[_node_out(node) for node in export.nodes],
        edges=[_edge_out(edge) for edge in export.edges],
        truncated=export.truncated,
    )


def _find_node(export: GraphExport, name: str) -> GraphExportNode | None:
    """Locate the entity a detail request asked for inside its own slice.

    Args:
        export: The depth-1 slice centred on ``name``.
        name: The requested entity name.

    Returns:
        The matching node, or ``None``. Exact first, then case-insensitive:
        the engine normalizes extracted entity names (upper-casing among
        them), so a link built from a citation's own spelling must still
        resolve.
    """
    for node in export.nodes:
        if node.id == name:
            return node
    folded = name.casefold()
    for node in export.nodes:
        if node.id.casefold() == folded:
            return node
    return None


def _transcript_refs(doc_keys: Sequence[str], workdir: Path) -> list[GraphTranscriptRefOut]:
    """Resolve an entity's document keys into source-day cards.

    The manifest is the durable record of what each key means (thread name,
    day span, message count), so the drill-down reads it rather than
    re-opening the engine. A key the manifest does not know still renders —
    parsed out of the key itself — because the graph *does* hold it and
    hiding it would be the bigger lie.

    Args:
        doc_keys: The node's document provenance.
        workdir: The engine's working directory (where the manifest lives).

    Returns:
        One card per key, in the engine's own order.
    """
    manifest = load_manifest(workdir)
    refs: list[GraphTranscriptRefOut] = []
    for key in doc_keys:
        thread_id, _, span = key.partition("::")
        known = manifest.docs.get(key)
        refs.append(
            GraphTranscriptRefOut(
                doc_key=key,
                thread_name=known.thread_name if known and known.thread_name else thread_id,
                span=known.span if known and known.span else span,
                message_count=len(known.message_guids) if known else 0,
            )
        )
    return refs


@router.get(
    "/api/graph/entities/{name:path}",
    responses={403: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
def graph_entity(name: str, service: ServiceDep) -> GraphEntityDetailOut:
    """Inspect one entity: its summary, its neighbourhood, its source days.

    The graph view's click-through (spec_graphrag §4.4): a name in the
    picture resolves to what the engine merged about it, the relations
    around it, and — the drill-down that makes the graph answerable — the
    transcript days it was extracted from.

    Args:
        name: The entity's name (path-encoded; extracted names may contain
            slashes). Canonical spelling resolves directly; any other casing
            resolves through a whole-graph lookup, because the engine's own
            label match is exact.
        service: The process-wide graph service.

    Returns:
        The entity with its depth-1 relations and its source days.

    Raises:
        HTTPException: ``403 graph_disabled`` when the kill switch is off;
            ``404 entity_not_found`` when the graph holds no such entity
            (which includes a graph that has not been built yet);
            ``503 graph_unavailable`` when the engine will not open.
    """
    _require_enabled()
    export = _read_graph(service, name, max_depth=1, max_nodes=MAX_EXPORT_NODES)
    node = _find_node(export, name) if name else None
    if node is None and name:
        # The engine's label lookup is exact, so a differently-cased name
        # comes back as an *empty* slice — the in-slice case fallback never
        # gets a candidate. Find the canonical spelling in the whole graph,
        # then re-slice on it. Honestly bounded: an entity outside the top
        # MAX_EXPORT_NODES by degree is reachable only by exact spelling.
        whole = _read_graph(service, _WHOLE_GRAPH, max_depth=1, max_nodes=MAX_EXPORT_NODES)
        canonical = _find_node(whole, name)
        if canonical is not None:
            export = _read_graph(service, canonical.id, max_depth=1, max_nodes=MAX_EXPORT_NODES)
            node = _find_node(export, canonical.id)
    if node is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "entity_not_found", "message": f"No graph entity named {name!r}"},
        )
    return GraphEntityDetailOut(
        entity=_node_out(node),
        relations=[
            _edge_out(edge) for edge in export.edges if node.id in (edge.source, edge.target)
        ],
        sources=_transcript_refs(node.doc_keys, service.workdir),
    )
