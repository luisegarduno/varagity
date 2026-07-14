"""Corpus document routes: upload, list, delete (spec_v2 §4.2).

``POST /api/documents`` writes multipart uploads into ``DOCS_PATH``
(validated against ``ALLOWED_EXTENSIONS`` + ``UPLOAD_MAX_MB``, **no
auto-ingest** — ingestion is its own explicit action). ``GET`` lists the
``documents`` table with each document's extraction mix. ``DELETE`` removes
a document's chunks from **both** stores — the v1 gap ("removing a file
does not remove its chunks") turned into GUI-driven GC — Elasticsearch
first, pgvector last, mirroring the reingest ordering rationale (the
``documents`` row is the idempotency marker, so it must go last); with
``?remove_file=true`` the source file is also unlinked when it lives inside
``DOCS_PATH``, so the next ingest can't resurrect the document.
"""

import contextlib
import logging
import shutil
from pathlib import Path
from typing import Annotated

from elastic_transport import TransportError
from fastapi import APIRouter, Depends, HTTPException, UploadFile

from varagity.api.deps import get_bm25_store, get_vector_store
from varagity.api.schemas import (
    DocumentDeleteResponse,
    DocumentOut,
    UploadedFileOut,
    UploadResponse,
)
from varagity.config import get_settings
from varagity.stores.bm25_store import ElasticsearchBM25
from varagity.stores.vector_store import ContextualVectorDB

logger = logging.getLogger(__name__)

router = APIRouter(tags=["documents"])

VectorStoreDep = Annotated[ContextualVectorDB, Depends(get_vector_store)]
BM25StoreDep = Annotated[ElasticsearchBM25, Depends(get_bm25_store)]

# Uploads stream to disk in slices so an oversized file aborts at the cap
# instead of buffering whole.
_COPY_CHUNK_BYTES = 1 << 20


def _safe_name(raw: str | None) -> str | None:
    """Reduce an upload's client-supplied name to a safe basename.

    Args:
        raw: The multipart filename as sent by the browser.

    Returns:
        The bare file name, or ``None`` when nothing safe remains (empty,
        path-only, or a dotfile-with-no-stem like ``".pdf"``).
    """
    if raw is None:
        return None
    name = Path(raw.replace("\\", "/")).name.strip()
    if not name or name.startswith(".") or name in {"..", "."}:
        return None
    return name


def _store_upload(upload: UploadFile, docs_root: Path, max_bytes: int) -> UploadedFileOut:
    """Validate and write one uploaded file into the corpus directory.

    Args:
        upload: The multipart part (spooled by Starlette).
        docs_root: The resolved ``DOCS_PATH`` directory.
        max_bytes: The per-file cap (``UPLOAD_MAX_MB`` in bytes).

    Returns:
        The per-file outcome (rejections are reported, never raised — one
        bad file must not fail the batch).
    """
    settings = get_settings()
    name = _safe_name(upload.filename)
    if name is None:
        return UploadedFileOut(
            file_name=upload.filename or "(unnamed)",
            size_bytes=0,
            stored=False,
            reason="invalid_filename",
        )
    extension = Path(name).suffix.lower()
    if extension not in settings.allowed_extension_set:
        return UploadedFileOut(
            file_name=name, size_bytes=0, stored=False, reason="extension_not_allowed"
        )

    target = docs_root / name
    replaced = target.exists()
    partial = target.with_name(f".{name}.upload-partial")
    written = 0
    try:
        with partial.open("wb") as sink:
            while chunk := upload.file.read(_COPY_CHUNK_BYTES):
                written += len(chunk)
                if written > max_bytes:
                    return UploadedFileOut(
                        file_name=name, size_bytes=0, stored=False, reason="file_too_large"
                    )
                sink.write(chunk)
        shutil.move(partial, target)
    except OSError as error:
        # A write failure is a server-side problem (an unwritable ./docs
        # mount, a full disk), never the file's fault — contained per file
        # so a mixed batch still reports coherently, and escalated to the
        # structured 500 by the route when nothing landed at all.
        logger.error("could not write upload %s under %s: %s", name, docs_root, error)
        return UploadedFileOut(file_name=name, size_bytes=0, stored=False, reason="write_failed")
    finally:
        with contextlib.suppress(OSError):  # best-effort cleanup on the same bad mount
            partial.unlink(missing_ok=True)
    logger.info("stored upload %s (%d bytes%s)", target, written, ", replaced" if replaced else "")
    return UploadedFileOut(file_name=name, size_bytes=written, stored=True, replaced=replaced)


@router.post("/api/documents", status_code=201)
def upload_documents(files: list[UploadFile]) -> UploadResponse:
    """Upload file(s) into ``DOCS_PATH`` (no auto-ingest).

    Each file is validated and stored independently; rejected files are
    reported per-file so a mixed batch partially succeeds. A same-named
    existing file is replaced (the re-upload flow) — its changed content
    hash makes the next ingest re-process it.

    Args:
        files: The multipart file parts.

    Returns:
        Per-file outcomes, in upload order.

    Raises:
        HTTPException: ``422 no_file_stored`` when every file was rejected
            on its own merits (extension/size/name); ``500
            docs_path_not_writable`` when nothing landed because the server
            couldn't write ``DOCS_PATH`` (e.g. the ``./docs`` bind mount is
            not writable by the api container's user).
    """
    settings = get_settings()
    docs_root = Path(settings.DOCS_PATH)
    not_writable = HTTPException(
        status_code=500,
        detail={
            "code": "docs_path_not_writable",
            "message": (
                f"the API cannot write DOCS_PATH ({docs_root}) — in compose, the "
                "./docs bind mount must be writable by the api container's user "
                "(rebuild the images if you changed uids; see the runbook)"
            ),
        },
    )
    try:
        docs_root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        logger.error("cannot create DOCS_PATH %s: %s", docs_root, error)
        raise not_writable from error
    max_bytes = settings.UPLOAD_MAX_MB * 1024 * 1024

    results = [_store_upload(upload, docs_root, max_bytes) for upload in files]
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
    return UploadResponse(files=results)


@router.get("/api/documents")
def list_documents(store: VectorStoreDep) -> list[DocumentOut]:
    """List every ingested document (the corpus-management table).

    Files uploaded but not yet ingested don't appear here — the ``documents``
    table records ingests; the GUI pairs this list with the upload outcomes
    it already holds.

    Args:
        store: The per-request vector store.

    Returns:
        One entry per document, newest ingest first.
    """
    return [
        DocumentOut(
            file_name=Path(info.source).name,
            **info.model_dump(),
        )
        for info in store.list_documents()
    ]


@router.delete("/api/documents/{doc_id}")
def delete_document(
    doc_id: str,
    store: VectorStoreDep,
    bm25: BM25StoreDep,
    remove_file: bool = False,
) -> DocumentDeleteResponse:
    """Remove a document's chunks from both stores (GUI-driven GC).

    Elasticsearch first, pgvector last: the pgvector ``documents`` row is
    the marker — if the ES delete fails, the document still lists and the
    delete can be retried; a marker deleted first would strand ES chunks
    invisibly (the exact v1 gap this route closes).

    Args:
        doc_id: The document to remove.
        store: The per-request vector store.
        bm25: The per-request BM25 store.
        remove_file: Also unlink the source file — honored only when it
            resolves inside ``DOCS_PATH`` (otherwise the next ingest would
            simply re-add the document).

    Returns:
        The deletion counts and whether the file was removed.

    Raises:
        HTTPException: ``404 document_not_found`` for an unknown id;
            ``503 es_unreachable`` when Elasticsearch is down (nothing
            deleted — retry when it returns).
    """
    documents = {info.doc_id: info for info in store.list_documents()}
    info = documents.get(doc_id)
    if info is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "document_not_found", "message": f"No document with id {doc_id!r}"},
        )

    try:
        bm25.delete_document(doc_id)
    except TransportError as error:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "es_unreachable",
                "message": f"elasticsearch unreachable — nothing deleted ({error})",
            },
        ) from error
    store.delete_document(doc_id)

    file_removed = False
    if remove_file:
        docs_root = Path(get_settings().DOCS_PATH).resolve()
        source = Path(info.source)
        try:
            inside_corpus = source.resolve().is_relative_to(docs_root)
        except OSError:  # unresolvable (dangling symlink target, …)
            inside_corpus = False
        if inside_corpus and source.exists():
            source.unlink()
            file_removed = True
        elif not inside_corpus:
            logger.warning(
                "not removing %s — outside DOCS_PATH (%s); the next ingest will re-add it",
                info.source,
                docs_root,
            )

    logger.info(
        "deleted document %s (%d chunk(s)) from both stores%s",
        doc_id,
        info.n_chunks,
        " + source file" if file_removed else "",
    )
    return DocumentDeleteResponse(
        doc_id=doc_id, chunks_deleted=info.n_chunks, file_removed=file_removed
    )
