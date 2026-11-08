"""Backup routes: list, export, status (SSE), download (ADR-019).

The ``/corpus`` "Export & restore" panel's whole server surface, mirroring
the ingest routes' shape — a ``202`` start, a status stream that replays the
backlog to late joiners, the structured error envelope:

* ``GET /api/backups/{corpus}`` lists the bundles in the corpus's reserved
  ``varagity-exports/`` folder, newest first, each with its ``bundle.json``
  (read alone through the zip's central directory). A file copied in by hand
  is listed too; an in-flight ``.partial`` never is.
* ``POST /api/backups/{corpus}/export`` starts an export run. Everything
  that can refuse it is checked first, so a refusal is a clean status, never
  a failed run: postgres unreachable (``503``), nothing ingested (``422
  nothing_to_export``), the folder unwritable (``500``), another operation
  holding the corpus (``409 corpus_busy``).
* ``GET /api/backups/{corpus}/status`` streams the run's ``status`` /
  ``progress`` / ``log`` events.
* ``GET /api/backups/{corpus}/files/{name}`` returns a bundle as an
  attachment — a plain navigation, so the browser downloads it with no
  CORS involved — resolving ``name`` inside the folder or answering
  ``404 bundle_not_found``.

**Why "save to the folder" rather than "open the folder".** No browser API
can open a folder on the user's machine, and the API runs in a container
that can't reach the desktop; the bundle lands where the ticket wants it
(beside the sources, inside the bind mount that survives ``down -v``), and
the panel shows that path with Copy-path and Download.

``corpus`` is a path parameter from the start; only ``rag`` is accepted
until the graph corpus gains its own export.
"""

import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent

from varagity.api.backup_runner import BackupRunner, get_backup_runners
from varagity.api.corpus_gate import CorpusBusy
from varagity.api.deps import get_backup_preflight, get_vector_store
from varagity.api.schemas import BackupRunOut, BundleManifestOut, BundleOut, ErrorResponse
from varagity.backup.bundle import (
    BundleListing,
    display_path,
    exports_dir,
    is_bundle_name,
    list_bundles,
)
from varagity.paths import resolve_contained
from varagity.stores.vector_store import ContextualVectorDB

logger = logging.getLogger(__name__)

router = APIRouter(tags=["backups"])

BackupCorpus = Literal["rag"]
"""The corpora with backups (the graph corpus joins in a later phase)."""

RunnersDep = Annotated[dict[str, BackupRunner], Depends(get_backup_runners)]
PreflightDep = Annotated[Callable[[str, str], Awaitable[None]], Depends(get_backup_preflight)]
VectorStoreDep = Annotated[ContextualVectorDB, Depends(get_vector_store)]

# The structured 500 code per corpus when its exports folder can't be
# written — the same codes the corpus's upload route uses for its root.
_NOT_WRITABLE_CODES = {"rag": "docs_path_not_writable", "graph": "graph_docs_path_not_writable"}


def _bundle_out(listing: BundleListing) -> BundleOut:
    """Render one listed bundle for the wire.

    Args:
        listing: The bundle file and its manifest (if readable).

    Returns:
        The ``BundleOut`` entry.
    """
    manifest = listing.manifest
    return BundleOut(
        name=listing.name,
        size_bytes=listing.size_bytes,
        modified_at=listing.modified_at,
        manifest=None if manifest is None else BundleManifestOut(**manifest.model_dump()),
    )


def _ensure_writable_exports_dir(corpus: BackupCorpus) -> Path:
    """Create the corpus's exports folder and confirm the API can write it.

    Args:
        corpus: The corpus about to be exported.

    Returns:
        The exports folder.

    Raises:
        HTTPException: ``500 docs_path_not_writable`` when the folder can't
            be created or written (in compose, the ``./docs`` bind mount
            must be writable by the api container's user — or a *file*
            named ``varagity-exports`` is squatting on the folder's name).
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


@router.get("/api/backups/{corpus}")
def list_backups(corpus: BackupCorpus) -> list[BundleOut]:
    """List the backup bundles in a corpus's ``varagity-exports/`` folder.

    A directory read plus one small member per bundle — the manifest is
    found through each zip's central directory, so a folder of large
    bundles lists in milliseconds.

    Args:
        corpus: The corpus whose backups to list.

    Returns:
        One entry per bundle file, newest first (empty when the folder does
        not exist yet).
    """
    return [_bundle_out(listing) for listing in list_bundles(exports_dir(corpus))]


@router.post(
    "/api/backups/{corpus}/export",
    status_code=202,
    responses={
        409: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def start_export(
    corpus: BackupCorpus,
    runners: RunnersDep,
    preflight: PreflightDep,
    store: VectorStoreDep,
) -> BackupRunOut:
    """Start a background export and return its handle.

    The bundle is written into ``<corpus root>/varagity-exports/`` as
    ``varagity-<corpus>-<UTC yyyymmdd-HHMM>.zip`` (a same-minute repeat gets
    a ``-2`` suffix — never a replace), from one consistent snapshot of the
    corpus.

    Args:
        corpus: The corpus to export.
        runners: The process-wide backup runners.
        preflight: The awaitable reachability check (structured 503).
        store: The per-request vector store (the nothing-to-export check).

    Returns:
        The new run's snapshot (state ``running``).

    Raises:
        HTTPException: ``503 postgres_unreachable`` when the database is
            down; ``422 nothing_to_export`` when nothing is ingested; ``500
            docs_path_not_writable`` when the exports folder can't be
            written; ``409 corpus_busy`` while any operation — another
            export included — holds the corpus (the message names it).
    """
    await preflight(corpus, "export")
    if await run_in_threadpool(store.document_count) == 0:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "nothing_to_export",
                "message": "the corpus holds no documents — ingest something before exporting",
            },
        )
    await run_in_threadpool(_ensure_writable_exports_dir, corpus)
    try:
        return runners[corpus].start("export")
    except CorpusBusy as error:
        raise HTTPException(
            status_code=409,
            detail={"code": "corpus_busy", "message": str(error)},
        ) from error


@router.get("/api/backups/{corpus}/status", response_class=EventSourceResponse)
async def backup_status(
    corpus: BackupCorpus, runners: RunnersDep
) -> AsyncIterator[ServerSentEvent]:
    """Stream the corpus's current (or last) backup run as SSE.

    Frames: ``status`` (snapshot; also terminal, with the summary and the
    saved path), ``progress`` (``snapshot`` → ``documents`` → ``chunks`` →
    ``finalize``), ``log`` (relayed ``varagity.backup`` lines). With no run
    in this API process the stream is a single idle ``status`` frame; after
    a terminal run it replays that run and closes.

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
        404: {"model": ErrorResponse},
    },
)
def download_backup(corpus: BackupCorpus, name: str) -> FileResponse:
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
        HTTPException: ``404 bundle_not_found`` when no such bundle lives in
            the folder — which covers a name trying to escape it.
    """
    path = _bundle_path(corpus, name)
    return FileResponse(path, media_type="application/zip", filename=name)
