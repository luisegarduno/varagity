"""The RAG bundle: every ``documents`` and ``chunks`` row, with its vector (ADR-019).

A RAG backup is the smallest thing that restores the chunk corpus with no
model call: the rows pgvector holds (contextual blurbs included) and their
embeddings. The BM25 index is deliberately absent — every field of an
Elasticsearch document is a copy of a ``chunks`` column, so a restore
rebuilds it from these same rows.

The bundle's data members, in write order::

    documents.jsonl   one documents row per line, plus its relative_path    deflated
    chunks.jsonl      one chunks row per line, every column but the vector  deflated
    embeddings.f32    N × EMBEDDING_DIM little-endian float32 — row i is    stored
                      the vector of chunks.jsonl line i

``relative_path`` is each document's ``source`` relative to ``DOCS_PATH``
(``null`` for a document ingested from elsewhere): ``doc_id`` already hashes
it, and it is what lets a restore re-root ``source`` onto whatever corpus
directory the restoring stack has. Everything is read from **one** corpus
snapshot (:meth:`~varagity.stores.vector_store.ContextualVectorDB.iter_corpus`),
so the three members always describe the same moment.

A **restore** (:func:`restore_rag`) turns a bundle back into rows without a
model call. Under the corpus's exclusive advisory lock, one Postgres
transaction replaces both tables — the bundle's lines streamed straight into
binary ``COPY``, each member's sha256 checked again as it is read — and its
commit is where the restore takes effect. The BM25 index is then rebuilt
from the restored rows. The corpus-stale flag it leaves comes from the
bundle (:func:`derive_corpus_stale`).
"""

import json
import logging
import tempfile
import time
from array import array
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import IO, Any, Literal

from pgvector import Vector
from pydantic import AwareDatetime, BaseModel, ValidationError

from varagity.backup.bundle import (
    BundleInvalid,
    BundleManifest,
    BundleSource,
    BundleWriter,
    MemberSink,
    WrittenBundle,
    inspect_bundle,
    open_bundle,
    write_bundle,
)
from varagity.config import get_settings
from varagity.paths import relative_to_root
from varagity.stores.bm25_store import ElasticsearchBM25
from varagity.stores.vector_store import (
    ContextualVectorDB,
    CorpusChunkRow,
    CorpusDocumentRow,
    CorpusSnapshot,
)

logger = logging.getLogger(__name__)

DOCUMENTS_MEMBER = "documents.jsonl"
CHUNKS_MEMBER = "chunks.jsonl"
EMBEDDINGS_MEMBER = "embeddings.f32"

ExportStage = Literal["snapshot", "documents", "chunks", "finalize"]
"""The export's progress stages, in order."""

ProgressCallback = Callable[[ExportStage, int | None, int | None], None]
"""``(stage, current, total)`` — how an export reports where it is."""

RestoreStage = Literal["validate", "load", "index", "bm25"]
"""The restore's progress stages, in order."""

RestoreProgressCallback = Callable[[RestoreStage, int | None, int | None], None]
"""``(stage, current, total)`` — how a restore reports where it is."""


# Progress frames per counted stage: enough for a smooth bar, few enough
# that a 100k-chunk export doesn't flood the status stream.
_PROGRESS_FRAMES = 50

# Member writes are batched: a zip member's per-write overhead (deflate
# flush bookkeeping, hashing) is worth amortizing over ~1 MiB.
_FLUSH_BYTES = 1 << 20

# The spool is copied into the bundle in slices this size.
_COPY_BYTES = 4 << 20


class NothingToExport(RuntimeError):
    """Raised when the snapshot holds no documents — an empty bundle is not a backup."""


class BM25RebuildFailed(RuntimeError):
    """The rows are restored, but the BM25 index could not be rebuilt.

    The Postgres commit already happened, so the restore took effect;
    restoring the same bundle again rebuilds the index (and rewrites the
    identical rows on the way).
    """


@dataclass(frozen=True)
class RagExport:
    """What one RAG export wrote.

    Attributes:
        bundle: The published bundle (name, path, size, manifest).
        wall_clock_s: Seconds from snapshot to publish.
    """

    bundle: WrittenBundle
    wall_clock_s: float

    @property
    def counts(self) -> dict[str, int]:
        """The bundle's row counts (``documents``, ``chunks``).

        Returns:
            The manifest's counts.
        """
        return dict(self.bundle.manifest.counts)


class _BufferedSink:
    """Batches small writes into ~1 MiB slices for one bundle member.

    Args:
        sink: The member to write to.
    """

    def __init__(self, sink: MemberSink | IO[bytes]) -> None:
        self._sink = sink
        self._buffer = bytearray()

    def write(self, data: bytes) -> None:
        """Queue bytes, flushing once a slice is full.

        Args:
            data: The next bytes of the member.
        """
        self._buffer += data
        if len(self._buffer) >= _FLUSH_BYTES:
            self.flush()

    def flush(self) -> None:
        """Write whatever is queued."""
        if self._buffer:
            self._sink.write(bytes(self._buffer))
            self._buffer.clear()


class _Progress:
    """Throttled ``(stage, current, total)`` reporting for one counted stage.

    Args:
        report: The export's or restore's progress callback.
        stage: The stage being counted.
        total: Its denominator.
    """

    def __init__(self, report: Callable[..., None], stage: str, total: int) -> None:
        self._report = report
        self._stage = stage
        self._total = total
        self._step = max(1, total // _PROGRESS_FRAMES)
        report(stage, 0, total)

    def tick(self, done: int) -> None:
        """Report ``done`` rows when it lands on a frame boundary.

        Args:
            done: Rows finished so far.
        """
        if done % self._step == 0 or done == self._total:
            self._report(self._stage, done, self._total)


def _json_line(record: Mapping[str, Any]) -> bytes:
    """Encode one JSONL line (compact, UTF-8, newline-terminated).

    Args:
        record: The row as a JSON object.

    Returns:
        The encoded line.
    """
    return (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def document_record(row: CorpusDocumentRow, docs_root: Path) -> dict[str, Any]:
    """Render one ``documents`` row as its ``documents.jsonl`` object.

    Args:
        row: The snapshot row.
        docs_root: The resolved ``DOCS_PATH`` the relative path is taken
            against.

    Returns:
        Every column, plus ``relative_path`` (``None`` when the source lives
        outside the corpus directory).
    """
    return {
        "doc_id": row.doc_id,
        "source": row.source,
        "relative_path": relative_to_root(row.source, docs_root),
        "file_type": row.file_type,
        "content_hash": row.content_hash,
        "n_chunks": row.n_chunks,
        "ingested_at": row.ingested_at.isoformat(),
    }


def chunk_record(row: CorpusChunkRow) -> dict[str, Any]:
    """Render one ``chunks`` row as its ``chunks.jsonl`` object.

    Args:
        row: The snapshot row.

    Returns:
        Every column except the embedding (which rides ``embeddings.f32``,
        at the same row index).
    """
    return {
        "chunk_id": row.chunk_id,
        "doc_id": row.doc_id,
        "original_index": row.original_index,
        "chunk_index": row.chunk_index,
        "content": row.content,
        "context": row.context,
        "contextualized_content": row.contextualized_content,
        "metadata": row.metadata,
        "created_at": row.created_at.isoformat(),
    }


def vector_bytes(vector: Vector, dim: int, chunk_id: str) -> bytes:
    """Encode one vector as a little-endian float32 matrix row.

    pgvector's binary form is a ``>HH`` header (dimensions, unused) and
    big-endian float4s; one byte swap turns that payload into the
    little-endian row the bundle stores, on any host, without a decimal
    round trip — the bits are the stored bits.

    Args:
        vector: The stored vector.
        dim: The dimension the bundle declares (``EMBEDDING_DIM``).
        chunk_id: The owning chunk (for the error message).

    Returns:
        ``dim × 4`` bytes.

    Raises:
        ValueError: If the vector's dimension isn't ``dim`` — the matrix
            would no longer be rectangular, and the manifest would lie.
    """
    raw = vector.to_binary()
    dimensions = int.from_bytes(raw[:2], "big")
    if dimensions != dim:
        raise ValueError(
            f"chunk {chunk_id} has a {dimensions}-dimensional embedding, but EMBEDDING_DIM is {dim}"
        )
    values = array("f")
    values.frombytes(raw[4:])
    values.byteswap()
    return values.tobytes()


def _write_documents(
    bundle: BundleWriter, snapshot: CorpusSnapshot, docs_root: Path, report: ProgressCallback
) -> int:
    """Write ``documents.jsonl``.

    Args:
        bundle: The bundle being filled.
        snapshot: The corpus snapshot.
        docs_root: The resolved ``DOCS_PATH``.
        report: The progress callback.

    Returns:
        Documents written.
    """
    progress = _Progress(report, "documents", snapshot.document_count)
    written = 0
    with bundle.member(DOCUMENTS_MEMBER) as member:
        sink = _BufferedSink(member)
        for row in snapshot.documents():
            sink.write(_json_line(document_record(row, docs_root)))
            written += 1
            progress.tick(written)
        sink.flush()
    return written


def _write_chunks(
    bundle: BundleWriter,
    snapshot: CorpusSnapshot,
    *,
    dim: int,
    spool: IO[bytes],
    report: ProgressCallback,
) -> tuple[int, Counter[str]]:
    """Write ``chunks.jsonl`` and spool the matching vectors, in one pass.

    A zip takes one member at a time, so the matrix can't stream beside the
    rows; it is spooled instead and copied in by :func:`_write_embeddings`.
    One pass over one snapshot is what makes row *i* of the matrix the
    vector of line *i* — by construction, not by re-sorting.

    Args:
        bundle: The bundle being filled.
        snapshot: The corpus snapshot.
        dim: ``EMBEDDING_DIM``.
        spool: A temporary file receiving the matrix rows.
        report: The progress callback.

    Returns:
        Chunks written, and a tally of every embedding model the rows'
        metadata names (a drifted corpus is worth a warning).
    """
    progress = _Progress(report, "chunks", snapshot.chunk_count)
    models: Counter[str] = Counter()
    written = 0
    vectors = _BufferedSink(spool)
    with bundle.member(CHUNKS_MEMBER) as member:
        sink = _BufferedSink(member)
        for row in snapshot.chunks():
            sink.write(_json_line(chunk_record(row)))
            vectors.write(vector_bytes(row.embedding, dim, row.chunk_id))
            models[str(row.metadata.get("embedding_model", "unknown"))] += 1
            written += 1
            progress.tick(written)
        sink.flush()
    vectors.flush()
    return written, models


def _write_embeddings(bundle: BundleWriter, spool: IO[bytes]) -> None:
    """Copy the spooled matrix into ``embeddings.f32`` (stored, not deflated).

    Args:
        bundle: The bundle being filled.
        spool: The temporary file :func:`_write_chunks` filled.
    """
    spool.seek(0)
    with bundle.member(EMBEDDINGS_MEMBER, compress=False) as member:
        while block := spool.read(_COPY_BYTES):
            member.write(block)


def export_rag(
    dest: Path,
    store: ContextualVectorDB,
    ingest_settings: Mapping[str, str],
    *,
    stale: bool,
    on_progress: ProgressCallback | None = None,
) -> RagExport:
    """Export the whole chunk corpus into a new bundle in ``dest``.

    Args:
        dest: The exports folder (``<DOCS_PATH>/varagity-exports``).
        store: The vector store to snapshot.
        ingest_settings: The reingest-affecting settings at export, in
            env-string form — the runner reads them, so this package never
            imports the API layer.
        stale: The corpus-stale flag at export.
        on_progress: Called with ``(stage, current, total)`` as the export
            moves through ``snapshot`` → ``documents`` → ``chunks`` →
            ``finalize``.

    Returns:
        The published bundle and how long it took.

    Raises:
        NothingToExport: If the snapshot holds no documents.
        ValueError: If a stored vector's dimension isn't ``EMBEDDING_DIM``.
        OSError: If the bundle can't be written.
    """
    settings = get_settings()
    report: ProgressCallback = on_progress if on_progress is not None else _ignore
    docs_root = Path(settings.DOCS_PATH).resolve()
    dim = settings.EMBEDDING_DIM
    started = time.perf_counter()
    created_at = datetime.now(UTC)

    report("snapshot", None, None)
    with store.iter_corpus() as snapshot:
        if snapshot.document_count == 0:
            raise NothingToExport("the corpus holds no documents — there is nothing to export")
        # File names only in these lines: they are relayed into the panel's
        # log, where an in-container path (/app/docs/…) would contradict the
        # checkout-relative "Saved to" path beside it.
        logger.info(
            "exporting %d document(s) and %d chunk(s) from one snapshot",
            snapshot.document_count,
            snapshot.chunk_count,
        )
        with write_bundle(dest, "rag", created_at=created_at) as bundle:
            documents = _write_documents(bundle, snapshot, docs_root, report)
            # The spool sits beside the bundle (same disk) and is hidden and
            # unnamed where the OS allows: it is never listed, and it
            # vanishes with the process if the export dies.
            with tempfile.TemporaryFile(dir=dest, prefix=".varagity-rag-spool-") as spool:
                chunks, models = _write_chunks(
                    bundle, snapshot, dim=dim, spool=spool, report=report
                )
                report("finalize", None, None)
                _write_embeddings(bundle, spool)
            bundle.seal(
                embedding_model=settings.EMBEDDING_MODEL,
                embedding_dim=dim,
                stale=stale,
                counts={"documents": documents, "chunks": chunks},
                ingest_settings=dict(ingest_settings),
            )
    _warn_on_model_drift(models, settings.EMBEDDING_MODEL)
    written = bundle.result
    wall_clock_s = time.perf_counter() - started
    logger.info(
        "exported %d document(s) and %d chunk(s) to %s (%d bytes) in %.1fs",
        documents,
        chunks,
        written.name,
        written.size_bytes,
        wall_clock_s,
    )
    return RagExport(bundle=written, wall_clock_s=wall_clock_s)


def _ignore(stage: str, current: int | None, total: int | None) -> None:
    """The do-nothing progress callback.

    Args:
        stage: Ignored.
        current: Ignored.
        total: Ignored.
    """


def _warn_on_model_drift(models: Counter[str], declared: str) -> None:
    """Warn when the rows name an embedding model the bundle doesn't declare.

    The manifest records ``EMBEDDING_MODEL``, which is what a restore checks;
    rows embedded under another model (an env change with no reingest) are
    exported faithfully but deserve a loud line in the log.

    Args:
        models: Chunks per ``metadata.embedding_model``.
        declared: The model the manifest declares.
    """
    others = {model: count for model, count in models.items() if model != declared}
    if others:
        listed = ", ".join(f"{model} ×{count}" for model, count in sorted(others.items()))
        logger.warning(
            "%d chunk(s) record a different embedding model than EMBEDDING_MODEL (%s): %s — "
            "the bundle declares %s",
            sum(others.values()),
            declared,
            listed,
            declared,
        )


# ── restore ────────────────────────────────────────────────────────────────


class _DocumentLine(BaseModel):
    """One ``documents.jsonl`` line, as a restore reads it."""

    doc_id: str
    source: str
    relative_path: str | None = None
    file_type: str
    content_hash: str
    n_chunks: int
    ingested_at: AwareDatetime


class _ChunkLine(BaseModel):
    """One ``chunks.jsonl`` line, as a restore reads it."""

    chunk_id: str
    doc_id: str
    original_index: int
    chunk_index: int
    content: str
    context: str | None
    contextualized_content: str
    metadata: dict[str, Any]
    created_at: AwareDatetime


@dataclass(frozen=True)
class RagRestore:
    """What one RAG restore did.

    Attributes:
        bundle_name: The restored bundle's file name.
        size_bytes: The bundle's size on disk.
        manifest: Its manifest.
        documents: ``documents`` rows written.
        chunks: ``chunks`` rows written.
        bm25_indexed: Chunks the rebuilt BM25 index holds.
        wall_clock_s: Seconds from validation to a searchable index.
        stale: The corpus-stale flag the restore left — derived from the
            bundle by the ``rag-restore`` flow; ``None`` until it has.
    """

    bundle_name: str
    size_bytes: int
    manifest: BundleManifest
    documents: int
    chunks: int
    bm25_indexed: int
    wall_clock_s: float
    stale: bool | None = None

    @property
    def counts(self) -> dict[str, int]:
        """What the restore wrote (``documents``, ``chunks``, ``bm25``).

        Returns:
            The row counts, plus the chunks the BM25 index now holds.
        """
        return {"documents": self.documents, "chunks": self.chunks, "bm25": self.bm25_indexed}


def restored_source(source: str, relative_path: str | None, docs_root: Path) -> str:
    """Re-root one document's ``source`` onto the restoring stack's corpus directory.

    ``doc_id`` already hashes the corpus-relative path, so the row's
    identity is portable; ``source`` (the absolute path the readers resolve
    — the document list, previews, "also remove the file") is rebuilt the
    way ingest builds it, ``(DOCS_PATH / relative_path).resolve()``. A
    document exported from outside the corpus has no relative path and
    keeps its original absolute ``source``.

    Args:
        source: The ``source`` recorded at export.
        relative_path: Its path relative to the exporting ``DOCS_PATH``, or
            ``None``.
        docs_root: The resolved ``DOCS_PATH`` of the restoring stack.

    Returns:
        The ``source`` to restore.

    Raises:
        BundleInvalid: If the relative path is absolute, empty, or climbs
            out of the corpus (``..``) — no export writes one.
    """
    if relative_path is None:
        return source
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise BundleInvalid(
            f"{DOCUMENTS_MEMBER} records relative_path {relative_path!r}, which is not a path "
            "inside the corpus."
        )
    return str(docs_root.joinpath(*relative.parts).resolve())


def ingest_drift(manifest: BundleManifest, current: Mapping[str, str]) -> list[str]:
    """Name the reingest-affecting settings a RAG bundle differs from this stack on.

    Args:
        manifest: The bundle's manifest.
        current: This stack's reingest-affecting settings, in env-string
            form (``varagity.api.backup_runner.current_ingest_settings``).

    Returns:
        The differing setting names, sorted — a setting recorded on only one
        side counts as differing, and a bundle that recorded no settings
        differs on all of them.
    """
    recorded = manifest.ingest_settings
    if recorded is None:
        return sorted(current)
    return sorted(
        name for name in set(recorded) | set(current) if recorded.get(name) != current.get(name)
    )


def derive_corpus_stale(manifest: BundleManifest, current: Mapping[str, str]) -> bool:
    """The corpus-stale flag a restored RAG bundle leaves.

    Stale when the bundle was stale at export, or when the settings its
    rows were made under differ from this stack's — the same "the corpus
    may not match the current settings" the flag always means.

    Args:
        manifest: The restored bundle's manifest.
        current: This stack's reingest-affecting settings, in env-string
            form.

    Returns:
        Whether the restored corpus is stale.
    """
    return manifest.stale or bool(ingest_drift(manifest, current))


def vector_from_bytes(raw: bytes, dim: int) -> Vector:
    """Decode one little-endian float32 matrix row back into a vector, bit-exact.

    The inverse of :func:`vector_bytes`: one byte swap turns the row into
    pgvector's big-endian payload, on any host.

    Args:
        raw: ``dim × 4`` bytes of ``embeddings.f32``.
        dim: The vector's dimension.

    Returns:
        The vector.
    """
    values = array("f")
    values.frombytes(raw)
    values.byteswap()
    return Vector.from_binary(dim.to_bytes(2, "big") + b"\x00\x00" + values.tobytes())


def _parse_line[LineT: (_DocumentLine, _ChunkLine)](
    model: type[LineT], line: bytes, member: str, number: int
) -> LineT:
    """Validate one JSONL line of a bundle member.

    Args:
        model: The line's shape.
        line: The raw line.
        member: The member it came from (for the message).
        number: Its 1-based line number.

    Returns:
        The validated line.

    Raises:
        BundleInvalid: If the line isn't a row of that shape.
    """
    try:
        return model.model_validate_json(line)
    except ValidationError as error:
        raise BundleInvalid(
            f"{member} line {number} is not a valid row ({error.error_count()} problem(s))."
        ) from error


def _document_rows(
    source: BundleSource, docs_root: Path, sources: dict[str, str], expected: int
) -> Iterator[CorpusDocumentRow]:
    """Stream ``documents.jsonl`` as restorable rows, re-rooted.

    Fills ``sources`` (``doc_id`` → restored ``source``) as it goes, which
    the chunk stream reads — :meth:`ContextualVectorDB.replace_corpus`
    consumes this stream whole before it starts on the chunks.

    Args:
        source: The open bundle.
        docs_root: The resolved ``DOCS_PATH``.
        sources: Filled with each document's restored ``source``.
        expected: The document count the manifest records.

    Yields:
        One row per line.

    Raises:
        BundleInvalid: For a malformed line, a count that disagrees with the
            manifest, or (on exhausting the member) a sha256 mismatch.
    """
    count = 0
    with source.member(DOCUMENTS_MEMBER) as lines:
        for number, line in enumerate(lines, start=1):
            record = _parse_line(_DocumentLine, line, DOCUMENTS_MEMBER, number)
            restored = restored_source(record.source, record.relative_path, docs_root)
            sources[record.doc_id] = restored
            count += 1
            yield CorpusDocumentRow(
                doc_id=record.doc_id,
                source=restored,
                file_type=record.file_type,
                content_hash=record.content_hash,
                n_chunks=record.n_chunks,
                ingested_at=record.ingested_at,
            )
    if count != expected:
        raise BundleInvalid(
            f"{DOCUMENTS_MEMBER} holds {count:,} document(s), but the manifest records "
            f"{expected:,}."
        )


def _chunk_rows(
    source: BundleSource,
    sources: Mapping[str, str],
    *,
    dim: int,
    expected: int,
    progress: _Progress,
) -> Iterator[CorpusChunkRow]:
    """Stream ``chunks.jsonl`` and ``embeddings.f32`` side by side as restorable rows.

    Line *i* takes row *i* of the matrix — the alignment the export wrote by
    construction — and its ``metadata.source`` takes its document's
    restored ``source``.

    Args:
        source: The open bundle.
        sources: ``doc_id`` → restored ``source`` (the document stream's).
        dim: The vectors' dimension.
        expected: The chunk count the manifest records.
        progress: The ``load`` stage's throttled ticks.

    Yields:
        One row per line, with its vector.

    Raises:
        BundleInvalid: For a malformed line, a chunk of no restored
            document, a matrix shorter or longer than the lines, a count that
            disagrees with the manifest, or (on exhausting the members) a
            sha256 mismatch.
    """
    row_bytes = dim * 4
    count = 0
    with source.member(CHUNKS_MEMBER) as lines, source.member(EMBEDDINGS_MEMBER) as vectors:
        for number, line in enumerate(lines, start=1):
            record = _parse_line(_ChunkLine, line, CHUNKS_MEMBER, number)
            document_source = sources.get(record.doc_id)
            if document_source is None:
                raise BundleInvalid(
                    f"{CHUNKS_MEMBER} line {number}: chunk {record.chunk_id} belongs to no "
                    f"document in {DOCUMENTS_MEMBER}."
                )
            raw = vectors.read(row_bytes)
            if len(raw) != row_bytes:
                raise BundleInvalid(
                    f"{EMBEDDINGS_MEMBER} ends before {CHUNKS_MEMBER} line {number} — every "
                    "chunk needs its vector."
                )
            metadata = dict(record.metadata)
            metadata["source"] = document_source
            yield CorpusChunkRow(
                chunk_id=record.chunk_id,
                doc_id=record.doc_id,
                original_index=record.original_index,
                chunk_index=record.chunk_index,
                content=record.content,
                context=record.context,
                contextualized_content=record.contextualized_content,
                metadata=metadata,
                created_at=record.created_at,
                embedding=vector_from_bytes(raw, dim),
            )
            count += 1
            progress.tick(count)
        if vectors.read(1):
            raise BundleInvalid(
                f"{EMBEDDINGS_MEMBER} holds more vectors than {CHUNKS_MEMBER} has lines."
            )
    if count != expected:
        raise BundleInvalid(
            f"{CHUNKS_MEMBER} holds {count:,} chunk(s), but the manifest records {expected:,}."
        )


def restore_rag(
    bundle: Path,
    store: ContextualVectorDB,
    bm25: ElasticsearchBM25,
    docs_root: Path,
    *,
    on_progress: RestoreProgressCallback | None = None,
) -> RagRestore:
    """Replace the chunk corpus with a bundle's rows, then rebuild the BM25 index.

    No model is called — nothing is parsed, chunked, contextualized or
    embedded. Under the corpus's **exclusive** advisory lock, held until the
    BM25 rebuild ends (so no ingest in any process can write rows the
    rebuild would miss, or touch the index while it is gone):

    1. ``validate`` — the cheap verdict again (the route verified the bundle
       before the run; this guards a direct call and a file swapped since).
    2. ``load`` — one transaction empties both tables and streams the
       bundle's lines into binary ``COPY``, re-rooting each ``source``;
       every member's sha256 is checked as it is read, so a bundle that
       changed after verification still can't commit.
    3. ``index`` — the HNSW index is rebuilt and the transaction commits:
       the restore has taken effect. Any failure before this rolled back,
       leaving the previous corpus untouched.
    4. ``bm25`` — the index is recreated and refilled from the restored
       rows.

    Args:
        bundle: The bundle file.
        store: The vector store to replace (its connection holds the lock).
        bm25: The BM25 store to rebuild.
        docs_root: The restoring stack's ``DOCS_PATH`` (re-rooting target).
        on_progress: Called with ``(stage, current, total)`` per stage; the
            ``load`` and ``bm25`` stages count chunks.

    Returns:
        What was restored, and how long it took.

    Raises:
        BundleInvalid: If the bundle is unsound or damaged (nothing was
            replaced).
        BundleIncompatible: If this stack must not restore it (nothing was
            replaced).
        varagity.stores.vector_store.CorpusLocked: If an ingest holds the
            corpus lock (nothing was replaced).
        BM25RebuildFailed: If the rows were restored but the BM25 index
            could not be rebuilt.
    """
    settings = get_settings()
    report: RestoreProgressCallback = on_progress if on_progress is not None else _ignore
    dim = settings.EMBEDDING_DIM
    root = docs_root.resolve()
    started = time.perf_counter()

    report("validate", None, None)
    manifest = inspect_bundle(bundle, "rag").manifest_or_raise()
    size_bytes = bundle.stat().st_size
    expected_documents = manifest.counts["documents"]
    expected_chunks = manifest.counts["chunks"]
    logger.info(
        "restoring %d document(s) and %d chunk(s) from %s",
        expected_documents,
        expected_chunks,
        bundle.name,
    )
    with store.corpus_lock(exclusive=True):
        sources: dict[str, str] = {}
        with open_bundle(bundle, manifest) as source:
            documents, chunks = store.replace_corpus(
                _document_rows(source, root, sources, expected_documents),
                _chunk_rows(
                    source,
                    sources,
                    dim=dim,
                    expected=expected_chunks,
                    progress=_Progress(report, "load", expected_chunks),
                ),
                on_indexing=lambda: report("index", None, None),
            )
        logger.info("loaded %d document(s) and %d chunk(s) — committed", documents, chunks)
        progress = _Progress(report, "bm25", chunks)
        try:
            with closing(store.iter_index_rows()) as rows:
                indexed = bm25.rebuild(rows, on_indexed=progress.tick)
        except Exception as error:
            raise BM25RebuildFailed(
                f"the rows are restored, but the BM25 index could not be rebuilt "
                f"({type(error).__name__}: {error}) — restore the same bundle again to "
                "repair it"
            ) from error
    wall_clock_s = time.perf_counter() - started
    logger.info(
        "restored %d document(s) and %d chunk(s) from %s in %.1fs; BM25 holds %d",
        documents,
        chunks,
        bundle.name,
        wall_clock_s,
        indexed,
    )
    return RagRestore(
        bundle_name=bundle.name,
        size_bytes=size_bytes,
        manifest=manifest,
        documents=documents,
        chunks=chunks,
        bm25_indexed=indexed,
        wall_clock_s=wall_clock_s,
    )
