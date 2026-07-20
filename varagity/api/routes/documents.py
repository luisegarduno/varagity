"""Corpus document routes: upload, list, delete (spec_v2 §4.2, spec_v3 §5).

``POST /api/documents`` writes multipart uploads into ``DOCS_PATH``
(validated against ``ALLOWED_EXTENSIONS`` + ``UPLOAD_MAX_MB``, **no
auto-ingest** — ingestion is its own explicit action). v3 adds optional
per-file relative ``paths`` (folder uploads): structure is *identity*, not
decoration — ``doc_id`` hashes the path relative to ``DOCS_PATH``, so
``q3/notes.md`` and ``q4/notes.md`` must land at distinct paths or the
second silently replaces the first. ``GET`` lists the ``documents`` table
with each document's extraction mix and its path relative to ``DOCS_PATH``
(the GUI's folder-grouping key). ``DELETE`` removes a document's chunks
from **both** stores — the v1 gap ("removing a file does not remove its
chunks") turned into GUI-driven GC — Elasticsearch first, pgvector last,
mirroring the reingest ordering rationale (the ``documents`` row is the
idempotency marker, so it must go last); with ``?remove_file=true`` the
source file is also unlinked when it lives inside ``DOCS_PATH``, so the
next ingest can't resurrect the document. ``POST /api/documents/delete``
is the same operation over a set (the corpus table's multi-select): the
same ordering applied set-wise, in one round trip per store instead of one
per document.

The ``…/preview/locate`` + ``…/preview/page/{page}`` pair (ADR-010) backs
the evidence panel's page preview: locate scores the source document's
pages for a chunk's text and returns highlight rects; the page route
serves the rendered PNG. Both degrade per-document (``available:false`` /
a 404 with the reason as its code) rather than 500 — the GUI falls back
to the full-text view.
"""

import contextlib
import logging
import re
import shutil
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Annotated

import pypdfium2 as pdfium
from elastic_transport import TransportError
from fastapi import APIRouter, Depends, Form, HTTPException, Response, UploadFile

from varagity.api.deps import get_bm25_store, get_vector_store
from varagity.api.schemas import (
    DocumentBulkDeleteRequest,
    DocumentBulkDeleteResponse,
    DocumentDeleteResponse,
    DocumentOut,
    PreviewLocateRequest,
    PreviewLocateResponse,
    PreviewRect,
    UploadedFileOut,
    UploadResponse,
)
from varagity.config import get_settings
from varagity.paths import resolve_contained
from varagity.preview import locate, render_page_png, resolve_preview_source
from varagity.stores.bm25_store import ElasticsearchBM25
from varagity.stores.records import DocumentInfo
from varagity.stores.vector_store import ContextualVectorDB

logger = logging.getLogger(__name__)

router = APIRouter(tags=["documents"])

VectorStoreDep = Annotated[ContextualVectorDB, Depends(get_vector_store)]
BM25StoreDep = Annotated[ElasticsearchBM25, Depends(get_bm25_store)]

# Uploads stream to disk in slices so an oversized file aborts at the cap
# instead of buffering whole.
_COPY_CHUNK_BYTES = 1 << 20

# Structural shape limits for relative upload paths (spec_v3 §5.2) —
# generous for real corpora, tight enough that hostile input can't smuggle
# absurdity past the per-segment rules. Depth is the configured cap
# (``UPLOAD_MAX_PATH_DEPTH``); these two are not worth configuring.
_MAX_PATH_CHARS = 1024
_MAX_SEGMENT_CHARS = 255

# A percent-hex triplet (%2e, %2F, …). Nothing here URL-decodes, but a path
# that *looks* encoded is rejected outright rather than stored literally —
# no later consumer should get the chance to decode it either.
_PERCENT_ESCAPE = re.compile(r"%[0-9a-fA-F]{2}")

# Windows reserved device names (matched on the pre-dot stem, per Windows
# semantics): a corpus directory later synced to Windows must not hold a
# file that cannot exist there.
_RESERVED_SEGMENTS = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{digit}" for digit in range(1, 10)}
    | {f"LPT{digit}" for digit in range(1, 10)}
)


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


def _safe_relative_path(raw: str) -> PurePosixPath | None:
    """Reduce a client-supplied relative path to a safe, contained one (spec_v3 §5.2).

    The layered defense, rules 1–5 of the spec (rule 6 — the ``resolve()``
    containment backstop — lives at the write site; rule 7 — the
    ``paths``/``files`` pairing contract — in the route):

    1. No absolute paths, no drive letters, no ``..`` segments.
    2. No empty segments, no ``.``-prefixed segments (dotfiles, ``.git/``),
       no Windows reserved device names.
    3. Every segment must already *be* its own sanitized basename (the
       :func:`_safe_name` rules applied per segment) — strictly: a segment
       sanitization would change is rejected, never transformed, so two
       lookalike paths can never collapse onto one target.
    4. Bounded shape: whole path ≤ 1024 characters, each segment ≤ 255
       (depth is the caller's check, against ``UPLOAD_MAX_PATH_DEPTH``,
       for its own ``path_too_deep`` reason).
    5. Extension vetting stays at the caller, unchanged from flat uploads —
       the final segment is the stored file name.

    Control characters (NUL included), percent-hex escapes (``%2e``), and
    any character that NFKC-normalizes to a path separator or colon are
    rejected outright: this route never decodes or normalizes them into
    effect, and nothing downstream should get the chance to.

    Args:
        raw: One client-supplied relative path (a ``paths[]`` form entry).

    Returns:
        The normalized relative path, or ``None`` when anything is off.
    """
    if not raw or len(raw) > _MAX_PATH_CHARS:
        return None
    if any(ord(char) < 32 or ord(char) == 127 for char in raw):
        return None
    text = raw.replace("\\", "/")
    if text.startswith("/"):
        return None
    segments = text.split("/")
    for segment in segments:
        if len(segment) > _MAX_SEGMENT_CHARS:
            return None
        if _safe_name(segment) != segment:
            return None
        folded = unicodedata.normalize("NFKC", segment)
        if any(char in folded for char in ("/", "\\", ":")):
            return None
        if _PERCENT_ESCAPE.search(segment):
            return None
        if segment.split(".")[0].upper() in _RESERVED_SEGMENTS:
            return None
    return PurePosixPath(*segments)


def _upload_size(upload: UploadFile) -> int:
    """Byte length of one spooled multipart part.

    Starlette fills ``UploadFile.size`` while parsing the body; the seek
    fallback covers parts constructed without it (tests, other servers).

    Args:
        upload: The multipart part.

    Returns:
        The part's size in bytes.
    """
    if upload.size is not None:
        return upload.size
    handle = upload.file
    position = handle.tell()
    handle.seek(0, 2)
    size = handle.tell()
    handle.seek(position)
    return size


def _store_upload(
    upload: UploadFile, docs_root: Path, max_bytes: int, raw_path: str | None = None
) -> UploadedFileOut:
    """Validate and write one uploaded file into the corpus directory.

    Args:
        upload: The multipart part (spooled by Starlette).
        docs_root: The resolved ``DOCS_PATH`` directory.
        max_bytes: The per-file cap (``UPLOAD_MAX_MB`` in bytes).
        raw_path: The client-declared relative path for this file (folder
            uploads, spec_v3 §5.2). ``None`` or ``""`` keeps the flat
            single-file contract byte-identical.

    Returns:
        The per-file outcome (rejections are reported, never raised — one
        bad file must not fail the batch).
    """
    settings = get_settings()
    relative: PurePosixPath | None = None
    if raw_path:  # None and "" both mean the flat contract
        relative = _safe_relative_path(raw_path)
        if relative is None:
            return UploadedFileOut(
                file_name=raw_path, size_bytes=0, stored=False, reason="invalid_path"
            )
        if len(relative.parts) > settings.UPLOAD_MAX_PATH_DEPTH:
            return UploadedFileOut(
                file_name=str(relative), size_bytes=0, stored=False, reason="path_too_deep"
            )
        name = relative.name
    else:
        flat_name = _safe_name(upload.filename)
        if flat_name is None:
            return UploadedFileOut(
                file_name=upload.filename or "(unnamed)",
                size_bytes=0,
                stored=False,
                reason="invalid_filename",
            )
        name = flat_name
    extension = Path(name).suffix.lower()
    if extension not in settings.allowed_extension_set:
        return UploadedFileOut(
            file_name=name, size_bytes=0, stored=False, reason="extension_not_allowed"
        )

    target = docs_root / name if relative is None else docs_root.joinpath(*relative.parts)
    if relative is not None:
        # Rule 6 — the containment backstop (spec_v3 §5.2): even if rules
        # 1–5 had a hole, or a symlink inside the corpus points out of it,
        # nothing is written outside DOCS_PATH. Mirrors the delete route's
        # check; OSError (a dangling symlink in the prefix) counts as
        # outside. docs_root is resolved at the call site, as it was inline.
        resolved = resolve_contained(target, docs_root.resolve())
        if resolved is None:
            return UploadedFileOut(
                file_name=raw_path or name, size_bytes=0, stored=False, reason="invalid_path"
            )
        target = resolved

    replaced = target.exists()
    partial = target.with_name(f".{name}.upload-partial")
    written = 0
    try:
        if relative is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
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
    return UploadedFileOut(
        file_name=name,
        size_bytes=written,
        stored=True,
        replaced=replaced,
        relative_path=relative.as_posix() if relative is not None else None,
    )


@router.post("/api/documents", status_code=201)
def upload_documents(
    files: list[UploadFile],
    paths: Annotated[list[str] | None, Form()] = None,
) -> UploadResponse:
    """Upload file(s) into ``DOCS_PATH`` (no auto-ingest).

    Each file is validated and stored independently; rejected files are
    reported per-file so a mixed batch partially succeeds. A same-named
    existing file is replaced (the re-upload flow) — its changed content
    hash makes the next ingest re-process it.

    With ``paths`` (folder uploads, spec_v3 §5.2), each file lands at its
    sanitized relative path under ``DOCS_PATH``, preserving nested
    structure — structure is *identity*: ``doc_id`` hashes the relative
    path, so ``q3/notes.md`` and ``q4/notes.md`` must not collapse onto one
    name. Entries pair with ``files`` positionally; an empty-string entry
    keeps that file on the flat path. Without ``paths`` the flat contract
    is untouched.

    Args:
        files: The multipart file parts.
        paths: Optional relative path per file, positionally aligned with
            ``files``.

    Returns:
        Per-file outcomes, in upload order.

    Raises:
        HTTPException: ``422 paths_mismatch`` when ``paths`` is present
            with a different length than ``files`` (a positional contract
            must be checked, not trusted); ``422 too_many_files`` / ``422
            batch_too_large`` when the batch busts ``UPLOAD_MAX_FILES`` /
            ``UPLOAD_MAX_TOTAL_MB`` (checked before anything is written);
            ``422 no_file_stored`` when every file was rejected on its own
            merits (extension/size/name/path); ``500
            docs_path_not_writable`` when nothing landed because the server
            couldn't write ``DOCS_PATH`` (e.g. the ``./docs`` bind mount is
            not writable by the api container's user).
    """
    settings = get_settings()
    if paths is not None and len(paths) != len(files):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "paths_mismatch",
                "message": f"paths carries {len(paths)} entries for {len(files)} files — "
                "the positional pairing must match exactly",
            },
        )
    if len(files) > settings.UPLOAD_MAX_FILES:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "too_many_files",
                "message": f"{len(files)} files exceed UPLOAD_MAX_FILES "
                f"({settings.UPLOAD_MAX_FILES}) — split the upload",
            },
        )
    total_bytes = sum(_upload_size(upload) for upload in files)
    if total_bytes > settings.UPLOAD_MAX_TOTAL_MB * 1024 * 1024:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "batch_too_large",
                "message": f"{total_bytes} bytes exceed UPLOAD_MAX_TOTAL_MB "
                f"({settings.UPLOAD_MAX_TOTAL_MB} MB) — split the upload",
            },
        )
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

    matched_paths: list[str | None] = list(paths) if paths is not None else [None] * len(files)
    results = [
        _store_upload(upload, docs_root, max_bytes, raw_path)
        for upload, raw_path in zip(files, matched_paths, strict=True)
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
    return UploadResponse(files=results)


def _relative_to_corpus(source: str, docs_root: Path) -> str | None:
    """A source's path relative to ``DOCS_PATH``, when it lives inside.

    The corpus table's grouping key: folder uploads (spec_v3 §5.2) keep
    their directory structure, and the GUI folds the flat document list
    back into those folders. Only the server knows the corpus root, so the
    relative path is computed here rather than string-guessed client-side.

    Args:
        source: The absolute source path recorded at ingest time.
        docs_root: The resolved ``DOCS_PATH`` directory.

    Returns:
        The POSIX-separated relative path, or ``None`` when the source
        doesn't resolve inside the corpus directory (ingested from
        elsewhere — such documents simply list flat).
    """
    resolved = resolve_contained(Path(source), docs_root)
    if resolved is None:
        return None
    return resolved.relative_to(docs_root).as_posix()


@router.get("/api/documents")
def list_documents(store: VectorStoreDep) -> list[DocumentOut]:
    """List every ingested document (the corpus-management table).

    Files uploaded but not yet ingested don't appear here — the ``documents``
    table records ingests; the GUI pairs this list with the upload outcomes
    it already holds. Each entry carries its ``relative_path`` under
    ``DOCS_PATH`` (``None`` for sources living elsewhere) so the GUI can
    group the table by the folders a folder upload created.

    Args:
        store: The per-request vector store.

    Returns:
        One entry per document, newest ingest first.
    """
    docs_root = Path(get_settings().DOCS_PATH).resolve()
    return [
        DocumentOut(
            file_name=Path(info.source).name,
            relative_path=_relative_to_corpus(info.source, docs_root),
            **info.model_dump(),
        )
        for info in store.list_documents()
    ]


def _prune_empty_parents(directory: Path, docs_root: Path) -> None:
    """Remove now-empty folders an unlinked source file leaves behind.

    Folder uploads create real directories under ``DOCS_PATH``; deleting a
    whole folder from the GUI unlinks its files one by one, and without
    this sweep the empty shells would accumulate invisibly (no document
    lists them, no ingest reads them). Walks upward from the file's
    folder, stopping at the first non-empty ancestor and never touching
    ``DOCS_PATH`` itself. Best-effort by design: ``rmdir`` refusing a
    non-empty directory (say, a concurrent upload into it) is the stop
    condition, not an error.

    Args:
        directory: The deleted file's (resolved) parent directory.
        docs_root: The resolved ``DOCS_PATH`` — the exclusive upper bound.
    """
    current = directory
    while current != docs_root and current.is_relative_to(docs_root):
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def _remove_source_file(info: DocumentInfo) -> bool:
    """Unlink one deleted document's source file, if it's ours to unlink.

    Honored only inside ``DOCS_PATH``: a document ingested from elsewhere
    would simply be re-added by the next ingest, and unlinking outside the
    corpus directory is not this route's business either way. Folders the
    unlink empties are pruned too, so deleting a folder upload's documents
    doesn't strand empty directory shells in the corpus.

    Args:
        info: The (already deleted) document's stored metadata.

    Returns:
        Whether the file was removed.
    """
    docs_root = Path(get_settings().DOCS_PATH).resolve()
    source = Path(info.source)
    resolved = resolve_contained(source, docs_root)
    if resolved is None:
        logger.warning(
            "not removing %s — outside DOCS_PATH (%s); the next ingest will re-add it",
            info.source,
            docs_root,
        )
        return False
    if not source.exists():
        return False
    source.unlink()
    _prune_empty_parents(resolved.parent, docs_root)
    return True


def _es_unreachable(error: TransportError) -> HTTPException:
    """Build the structured 503 for a downed Elasticsearch.

    Args:
        error: The transport failure raised by the BM25 store.

    Returns:
        The ``503 es_unreachable`` to raise — nothing has been deleted from
        either store, so the caller can simply retry.
    """
    return HTTPException(
        status_code=503,
        detail={
            "code": "es_unreachable",
            "message": f"elasticsearch unreachable — nothing deleted ({error})",
        },
    )


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
        raise _es_unreachable(error) from error
    store.delete_document(doc_id)

    file_removed = _remove_source_file(info) if remove_file else False
    logger.info(
        "deleted document %s (%d chunk(s)) from both stores%s",
        doc_id,
        info.n_chunks,
        " + source file" if file_removed else "",
    )
    return DocumentDeleteResponse(
        doc_id=doc_id, chunks_deleted=info.n_chunks, file_removed=file_removed
    )


@router.post("/api/documents/delete")
def delete_documents(
    payload: DocumentBulkDeleteRequest,
    store: VectorStoreDep,
    bm25: BM25StoreDep,
    remove_file: bool = False,
) -> DocumentBulkDeleteResponse:
    """Remove several documents' chunks from both stores (bulk GUI-driven GC).

    The corpus table's multi-select delete (spec_v2 §4.2). A ``POST``
    sub-path rather than a body on ``DELETE`` — the set to delete is the
    request's subject, and bodies on ``DELETE`` are the kind of thing
    intermediaries feel free to drop.

    Same store ordering as the single-document delete, applied set-wise:
    one ``terms`` ``delete_by_query`` against Elasticsearch first, one
    ``DELETE … WHERE doc_id = ANY(…)`` against pgvector last. So the
    invariant holds for the batch exactly as it does for one document — if
    Elasticsearch fails, no marker row is gone, every document still lists,
    and the whole batch is retryable. Unknown ids are reported in
    ``not_found`` rather than raising: a concurrent delete must not fail
    the rest of the batch.

    Args:
        payload: The requested ``doc_ids`` (duplicates collapse; order is
            preserved).
        store: The per-request vector store.
        bm25: The per-request BM25 store.
        remove_file: Also unlink each source file — honored per document,
            only inside ``DOCS_PATH``.

    Returns:
        Per-document outcomes plus the ids that had no ``documents`` row.

    Raises:
        HTTPException: ``503 es_unreachable`` when Elasticsearch is down
            (nothing deleted — retry when it returns).
    """
    documents = {info.doc_id: info for info in store.list_documents()}
    requested = list(dict.fromkeys(payload.doc_ids))  # collapse dupes, keep order
    targets = [documents[doc_id] for doc_id in requested if doc_id in documents]
    not_found = [doc_id for doc_id in requested if doc_id not in documents]

    doc_ids = [info.doc_id for info in targets]
    try:
        bm25.delete_documents(doc_ids)
    except TransportError as error:
        raise _es_unreachable(error) from error
    store.delete_documents(doc_ids)

    deleted = [
        DocumentDeleteResponse(
            doc_id=info.doc_id,
            chunks_deleted=info.n_chunks,
            file_removed=_remove_source_file(info) if remove_file else False,
        )
        for info in targets
    ]
    logger.info(
        "deleted %d document(s) (%d chunk(s)) from both stores%s%s",
        len(deleted),
        sum(entry.chunks_deleted for entry in deleted),
        f" + {sum(entry.file_removed for entry in deleted)} source file(s)" if remove_file else "",
        f"; {len(not_found)} unknown id(s)" if not_found else "",
    )
    return DocumentBulkDeleteResponse(deleted=deleted, not_found=not_found)


def _get_document_or_404(store: ContextualVectorDB, doc_id: str) -> DocumentInfo:
    """Fetch one document row or raise the structured 404.

    Args:
        store: The per-request vector store.
        doc_id: The document's stable id.

    Returns:
        The document's stored metadata.

    Raises:
        HTTPException: ``404 document_not_found`` for an unknown id.
    """
    info = store.get_document(doc_id)
    if info is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "document_not_found", "message": f"No document with id {doc_id!r}"},
        )
    return info


@router.post("/api/documents/{doc_id}/preview/locate")
def locate_preview(
    doc_id: str, payload: PreviewLocateRequest, store: VectorStoreDep
) -> PreviewLocateResponse:
    """Find the source page containing a chunk's text, with highlight rects.

    The evidence panel's locate step (ADR-010): resolve the document to an
    openable PDF (containment + content-hash verified; PPTX via the cached
    conversion), score its pages by word coverage, and return the best page
    plus normalized highlight rectangles. Every degradable condition — kill
    switch, unsupported format, missing/edited file, unavailable converter,
    no textual match — answers ``200 available:false`` with its reason; the
    GUI falls back to the full-text view.

    Args:
        doc_id: The chunk's parent document.
        payload: The chunk text to locate.
        store: The per-request vector store.

    Returns:
        The located page, rects, and coverage — or the degrade reason.

    Raises:
        HTTPException: ``404 document_not_found`` for an unknown id (the
            one non-degradable condition: there is nothing to fall back
            *to* — the GUI shows full text without asking again).
    """
    info = _get_document_or_404(store, doc_id)
    settings = get_settings()
    resolved = resolve_preview_source(info, settings)
    if resolved.pdf_path is None:
        return PreviewLocateResponse(available=False, reason=resolved.reason)
    try:
        result = locate(resolved.pdf_path, payload.text, min_coverage=settings.PREVIEW_MIN_COVERAGE)
    except pdfium.PdfiumError as error:
        # A corrupt/unopenable file must degrade, not 500 (the converted
        # artifact is covered too — hence the conversion-flavored reason).
        logger.warning("preview locate could not open %s (%s): %s", doc_id, info.source, error)
        return PreviewLocateResponse(available=False, reason="conversion_failed")
    if result.page is None:
        return PreviewLocateResponse(
            available=False,
            reason="no_match",
            page_count=result.page_count,
            coverage=result.coverage,
        )
    return PreviewLocateResponse(
        available=True,
        page=result.page,
        page_count=result.page_count,
        rects=[PreviewRect(x0=x0, y0=y0, x1=x1, y1=y1) for x0, y0, x1, y1 in result.rects],
        coverage=result.coverage,
    )


@router.get("/api/documents/{doc_id}/preview/page/{page}")
def preview_page(doc_id: str, page: int, store: VectorStoreDep) -> Response:
    """Serve one rendered page of a source document as a PNG.

    The image half of the preview pair (ADR-010), immutable-cacheable:
    ``doc_id`` is content-hashed and the resolution step re-verifies the
    on-disk ``content_hash`` before rendering, so a drifted file 404s
    (``file_changed``) rather than serving a lying image. Degradable
    reasons are 404 codes here — an ``<img>`` can't read a JSON envelope,
    and the client only requests pages a successful locate named.

    Args:
        doc_id: The document to render.
        page: 1-based page number (a locate response's ``page``).
        store: The per-request vector store.

    Returns:
        The PNG at ``PREVIEW_RENDER_WIDTH`` pixels wide, marked
        ``Cache-Control: public, max-age=31536000, immutable``.

    Raises:
        HTTPException: ``404 document_not_found`` for an unknown id; ``404``
            with the degrade reason as its code (``preview_disabled`` |
            ``unsupported_type`` | ``file_missing`` | ``file_changed`` |
            ``conversion_unavailable`` | ``conversion_failed``) when the
            document can't resolve; ``404 page_out_of_range`` beyond the
            last page.
    """
    info = _get_document_or_404(store, doc_id)
    settings = get_settings()
    resolved = resolve_preview_source(info, settings)
    if resolved.pdf_path is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": str(resolved.reason),
                "message": f"no previewable rendition of {doc_id!r} ({resolved.reason})",
            },
        )
    try:
        png = render_page_png(resolved.pdf_path, page, width=settings.PREVIEW_RENDER_WIDTH)
    except IndexError as error:
        raise HTTPException(
            status_code=404,
            detail={"code": "page_out_of_range", "message": str(error)},
        ) from error
    except pdfium.PdfiumError as error:
        logger.warning("preview render could not open %s (%s): %s", doc_id, info.source, error)
        raise HTTPException(
            status_code=404,
            detail={"code": "conversion_failed", "message": f"could not render {doc_id!r}"},
        ) from error
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )
