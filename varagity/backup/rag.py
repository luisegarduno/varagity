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
"""

import json
import logging
import tempfile
import time
from array import array
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any, Literal

from pgvector import Vector

from varagity.backup.bundle import BundleWriter, MemberSink, WrittenBundle, write_bundle
from varagity.config import get_settings
from varagity.paths import relative_to_root
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
        report: The export's progress callback.
        stage: The stage being counted.
        total: Its denominator.
    """

    def __init__(self, report: ProgressCallback, stage: ExportStage, total: int) -> None:
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


def _ignore(stage: ExportStage, current: int | None, total: int | None) -> None:
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
