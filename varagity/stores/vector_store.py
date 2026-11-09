"""``ContextualVectorDB`` — the pgvector-backed dense vector store.

The Anthropic cookbook's in-memory ``ContextualVectorDB`` responsibilities
(store vectors + metadata, cosine ``search``), re-backed with
PostgreSQL/pgvector (spec §11.2) so the index is durable, concurrent, and
inspectable via SQL. The schema itself lives in ``schema.sql`` (applied by
the postgres container on first boot).

It also owns the RAG corpus's **advisory lock** (ADR-019): one fixed key in
the database's advisory-lock namespace that every ingest holds *shared* for
its whole run and a backup restore holds *exclusive* from before it empties
the tables until the BM25 index is rebuilt. The API's in-process corpus gate
cannot see a CLI ingest (``main.py ingest`` / ``chat`` run in their own
process); this lock can, because it lives in the one place every writer
connects to. Ingests still overlap each other exactly as before, and a
session-level lock dies with its connection, so a crashed process never
strands the corpus.
"""

import logging
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.types.json import Json

from varagity.config import get_settings
from varagity.debug.show import check_verbose
from varagity.stores.base import ClosingContextMixin
from varagity.stores.records import ChunkRecord, DocumentInfo, RetrievedChunk

logger = logging.getLogger(__name__)

_UPSERT_DOCUMENT_SQL = """
INSERT INTO documents (doc_id, source, file_type, content_hash, n_chunks, ingested_at)
VALUES (%(doc_id)s, %(source)s, %(file_type)s, %(content_hash)s, %(n_chunks)s, now())
ON CONFLICT (doc_id) DO UPDATE SET
    source = EXCLUDED.source,
    file_type = EXCLUDED.file_type,
    content_hash = EXCLUDED.content_hash,
    n_chunks = EXCLUDED.n_chunks,
    ingested_at = now()
"""

_UPSERT_CHUNK_SQL = """
INSERT INTO chunks (chunk_id, doc_id, original_index, chunk_index, content, context,
                    contextualized_content, embedding, metadata, created_at)
VALUES (%(chunk_id)s, %(doc_id)s, %(original_index)s, %(chunk_index)s, %(content)s,
        %(context)s, %(contextualized_content)s, %(embedding)s, %(metadata)s, %(created_at)s)
ON CONFLICT (chunk_id) DO UPDATE SET
    doc_id = EXCLUDED.doc_id,
    original_index = EXCLUDED.original_index,
    chunk_index = EXCLUDED.chunk_index,
    content = EXCLUDED.content,
    context = EXCLUDED.context,
    contextualized_content = EXCLUDED.contextualized_content,
    embedding = EXCLUDED.embedding,
    metadata = EXCLUDED.metadata,
    created_at = EXCLUDED.created_at
"""

# Spec §11.2: cosine distance ordering; similarity score = 1 - distance.
_SEARCH_SQL = """
SELECT chunk_id, doc_id, original_index, content, context, metadata,
       1 - (embedding <=> %(qvec)s) AS score
FROM chunks
ORDER BY embedding <=> %(qvec)s
LIMIT %(k)s
"""

# Hydration for bm25/hybrid retrieval (spec §11.4): fetch full rows by the
# fusion identity. unnest() pairs the two arrays positionally.
_FETCH_BY_IDENTITY_SQL = """
SELECT c.chunk_id, c.doc_id, c.original_index, c.content, c.context, c.metadata
FROM chunks c
JOIN unnest(%(doc_ids)s::text[], %(original_indexes)s::int[])
     AS want(doc_id, original_index)
  ON c.doc_id = want.doc_id AND c.original_index = want.original_index
"""

# One document's chunks in reading order (the eval harness's fact-anchored
# golden resolution scans these — spec_v2 §7.4 chunker sweep).
_DOCUMENT_CHUNKS_SQL = """
SELECT chunk_id, doc_id, original_index, content, context, metadata
FROM chunks
WHERE doc_id = %(doc_id)s
ORDER BY chunk_index
"""

# The corpus-management list (spec_v2 §4.2 GET /api/documents): every
# documents row with its chunks' extraction mix, newest ingest first. The
# FILTER keeps 0-chunk documents (no extractable text) in the list with an
# empty mix instead of a {null: 0} artifact.
_LIST_DOCUMENTS_SQL = """
SELECT d.doc_id, d.source, d.file_type, d.content_hash, d.n_chunks, d.ingested_at,
       COALESCE(
           jsonb_object_agg(e.extraction, e.cnt) FILTER (WHERE e.extraction IS NOT NULL),
           '{}'::jsonb
       ) AS extraction_mix
FROM documents d
LEFT JOIN (
    SELECT doc_id, COALESCE(metadata->>'extraction', 'text') AS extraction, count(*) AS cnt
    FROM chunks
    GROUP BY doc_id, COALESCE(metadata->>'extraction', 'text')
) e USING (doc_id)
GROUP BY d.doc_id, d.source, d.file_type, d.content_hash, d.n_chunks, d.ingested_at
ORDER BY d.ingested_at DESC, d.source
"""

# One document by id, same shape as the list (ADR-010): the preview routes
# resolve doc_id → source/content_hash per request and must not pay a
# full-corpus list to do it.
_GET_DOCUMENT_SQL = """
SELECT d.doc_id, d.source, d.file_type, d.content_hash, d.n_chunks, d.ingested_at,
       COALESCE(
           jsonb_object_agg(e.extraction, e.cnt) FILTER (WHERE e.extraction IS NOT NULL),
           '{}'::jsonb
       ) AS extraction_mix
FROM documents d
LEFT JOIN (
    SELECT doc_id, COALESCE(metadata->>'extraction', 'text') AS extraction, count(*) AS cnt
    FROM chunks
    WHERE doc_id = %(doc_id)s
    GROUP BY doc_id, COALESCE(metadata->>'extraction', 'text')
) e USING (doc_id)
WHERE d.doc_id = %(doc_id)s
GROUP BY d.doc_id, d.source, d.file_type, d.content_hash, d.n_chunks, d.ingested_at
"""

# The corpus gauges (spec_v3 §6.1a): store state, not process history, so
# the Ingestion dashboard survives an api restart and sees CLI ingests too.
_DOCUMENTS_BY_TYPE_SQL = """
SELECT file_type, count(*)
FROM documents
GROUP BY file_type
"""

# chunking_strategy is a ChunkRecord field inside the metadata JSONB, not a
# chunks column (spec_v3 §6.1 says "column"; it is not). COALESCE mirrors
# _LIST_DOCUMENTS_SQL's extraction handling: rows written before the field
# existed group under 'unknown' rather than vanishing into a null label.
_CHUNKS_BY_STRATEGY_SQL = """
SELECT COALESCE(metadata->>'chunking_strategy', 'unknown'), count(*)
FROM chunks
GROUP BY 1
"""

# The corpus snapshot behind a backup export (ADR-019). The isolation level
# must be the transaction's first statement; READ ONLY because an export
# never writes, and so a bug can't turn the snapshot into a mutation.
_SNAPSHOT_ISOLATION_SQL = "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"

_CORPUS_COUNTS_SQL = "SELECT (SELECT count(*) FROM documents), (SELECT count(*) FROM chunks)"

# doc_id order keeps a bundle's documents.jsonl deterministic.
_CORPUS_DOCUMENTS_SQL = """
SELECT doc_id, source, file_type, content_hash, n_chunks, ingested_at
FROM documents
ORDER BY doc_id
"""

# original_index is the corpus-global ingest counter, so this is ingest order;
# doc_id breaks the (theoretical) tie between two concurrent runs' ranges —
# (doc_id, original_index) is unique, so the order is total and repeatable.
_CORPUS_CHUNKS_SQL = """
SELECT chunk_id, doc_id, original_index, chunk_index, content, context,
       contextualized_content, metadata, created_at, embedding
FROM chunks
ORDER BY original_index, doc_id
"""

# Rows fetched per server-side round trip while streaming a snapshot: a chunk
# row is a few KB with its vector, so a batch stays well under 10 MB.
_SNAPSHOT_BATCH_ROWS = 500

# The RAG corpus's advisory-lock key: b"VARAGITY" read as one big-endian
# bigint. Session-level locks on it are released by the session's end, so a
# crashed process can never strand the corpus.
CORPUS_LOCK_KEY = int.from_bytes(b"VARAGITY", "big")

# exclusive → the pg function (try-lock never waits: a busy corpus is an
# answer, not a queue).
_TRY_LOCK_SQL = {
    True: "SELECT pg_try_advisory_lock(%s)",
    False: "SELECT pg_try_advisory_lock_shared(%s)",
}
_UNLOCK_SQL = {
    True: "SELECT pg_advisory_unlock(%s)",
    False: "SELECT pg_advisory_unlock_shared(%s)",
}

# How long a restore waits to empty the tables: TRUNCATE needs ACCESS
# EXCLUSIVE, so it queues behind every open reader. Readers hold it for
# milliseconds; anything still there after this (an idle-in-transaction
# session) fails the restore cleanly instead of stalling it forever. SET
# LOCAL: it ends with the restore's transaction.
_RESTORE_LOCK_TIMEOUT_SQL = "SET LOCAL lock_timeout = '30s'"

_TRUNCATE_CORPUS_SQL = "TRUNCATE documents, chunks"

# The vector index is dropped for the load and built once after it (the
# pgvector README: "it's faster to create an index after loading your
# initial data"), from its own catalog definition so a tuned index (m,
# ef_construction) comes back as it was.
_HNSW_INDEX_NAME = "chunks_embedding_hnsw"
_INDEX_DEFINITION_SQL = """
SELECT indexdef FROM pg_indexes
WHERE schemaname = current_schema() AND indexname = 'chunks_embedding_hnsw'
"""
_DROP_HNSW_SQL = "DROP INDEX chunks_embedding_hnsw"
# schema.sql's own definition, for a database that has lost the index.
_DEFAULT_HNSW_SQL = (
    "CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)"
)

# Binary COPY: every value crosses the wire in its native binary form — the
# vectors as pgvector's own (dimension header + float4s), bit for bit.
_COPY_DOCUMENTS_SQL = (
    "COPY documents (doc_id, source, file_type, content_hash, n_chunks, ingested_at) "
    "FROM STDIN (FORMAT binary)"
)
_DOCUMENT_COPY_TYPES = ["text", "text", "text", "text", "int4", "timestamptz"]
_COPY_CHUNKS_SQL = (
    "COPY chunks (chunk_id, doc_id, original_index, chunk_index, content, context, "
    "contextualized_content, embedding, metadata, created_at) FROM STDIN (FORMAT binary)"
)
_CHUNK_COPY_TYPES = [
    "text",
    "text",
    "int4",
    "int4",
    "text",
    "text",
    "text",
    "vector",
    "jsonb",
    "timestamptz",
]

_ANALYZE_SQL = ("ANALYZE documents", "ANALYZE chunks")

# What the BM25 index mirrors of each chunk (bm25_store's five fields).
_INDEX_ROWS_SQL = """
SELECT chunk_id, doc_id, original_index, content, contextualized_content
FROM chunks
ORDER BY original_index, doc_id
"""


class CorpusLocked(RuntimeError):
    """Raised when the corpus lock is held in a mode that excludes the request.

    Attributes:
        exclusive: What the refused request wanted — ``True`` for a restore
            (refused by an ingest's shared hold), ``False`` for an ingest
            (refused by a restore's exclusive hold).
    """

    def __init__(self, *, exclusive: bool) -> None:
        """Word the refusal for whoever was refused.

        Args:
            exclusive: Whether the refused request wanted the lock
                exclusively.
        """
        super().__init__(
            "an ingest is running in another process (it holds the corpus lock) — "
            "try again when it finishes"
            if exclusive
            else "the corpus is being restored — try again when the restore finishes"
        )
        self.exclusive = exclusive


class CorpusTablesBusy(RuntimeError):
    """Raised when a restore can't get the corpus tables to itself in time.

    Some session kept ``documents``/``chunks`` open past the restore's lock
    timeout (an idle-in-transaction connection, a long dump). Nothing was
    replaced: the restore's transaction rolled back.
    """


@dataclass(frozen=True, slots=True)
class CorpusDocumentRow:
    """One ``documents`` row, every column, as a corpus snapshot reads it.

    Attributes:
        doc_id: The document's stable id.
        source: Absolute file path recorded at ingest time.
        file_type: File extension without the dot.
        content_hash: sha256 of the source file's bytes at ingest time.
        n_chunks: Chunks ingested (``0`` = no extractable text).
        ingested_at: When the document (last) landed in the stores.
    """

    doc_id: str
    source: str
    file_type: str
    content_hash: str
    n_chunks: int
    ingested_at: datetime


@dataclass(frozen=True, slots=True)
class CorpusChunkRow:
    """One ``chunks`` row, every column, as a corpus snapshot reads it.

    Attributes:
        chunk_id: ``{doc_id}::{chunk_index}``.
        doc_id: The owning document.
        original_index: The corpus-global index (the cross-store join key).
        chunk_index: Position within the document.
        content: The original chunk text.
        context: The contextual blurb (``None`` when contextualization was
            off).
        contextualized_content: The embedded and BM25-indexed text.
        metadata: The full ``ChunkRecord`` JSON.
        created_at: The chunk's creation timestamp.
        embedding: The passage vector, exactly as stored (read in pgvector's
            binary format, so no decimal round trip touches a bit).
    """

    chunk_id: str
    doc_id: str
    original_index: int
    chunk_index: int
    content: str
    context: str | None
    contextualized_content: str
    metadata: dict[str, Any]
    created_at: datetime
    embedding: Vector


@dataclass(frozen=True, slots=True)
class CorpusIndexRow:
    """What the BM25 index mirrors of one chunk — its five fields.

    Attributes:
        chunk_id: ``{doc_id}::{chunk_index}`` (the Elasticsearch ``_id``).
        doc_id: The owning document.
        original_index: The corpus-global index (the fusion join key).
        content: The original chunk text.
        contextualized_content: The embedded and BM25-indexed text.
    """

    chunk_id: str
    doc_id: str
    original_index: int
    content: str
    contextualized_content: str


class CorpusSnapshot:
    """One consistent, read-only view of ``documents`` and ``chunks``.

    Handed out by :meth:`ContextualVectorDB.iter_corpus` for the life of its
    ``with`` block: the counts and both row streams come from the same
    REPEATABLE READ transaction, so an ingest or delete committing mid-export
    can never tear a bundle — every row belongs to one moment, and the
    counts agree with the streams.

    Attributes:
        document_count: ``documents`` rows in the snapshot.
        chunk_count: ``chunks`` rows in the snapshot.
    """

    def __init__(
        self, conn: psycopg.Connection[Any], *, document_count: int, chunk_count: int
    ) -> None:
        """Bind the snapshot to its open transaction.

        Args:
            conn: The connection holding the snapshot's transaction.
            document_count: ``documents`` rows in the snapshot.
            chunk_count: ``chunks`` rows in the snapshot.
        """
        self._conn = conn
        self.document_count = document_count
        self.chunk_count = chunk_count

    def documents(self) -> Iterator[CorpusDocumentRow]:
        """Stream every ``documents`` row, ``doc_id`` order.

        Yields:
            One row per document.
        """
        for row in self._stream("varagity_snapshot_documents", _CORPUS_DOCUMENTS_SQL):
            yield CorpusDocumentRow(*row)

    def chunks(self) -> Iterator[CorpusChunkRow]:
        """Stream every ``chunks`` row with its vector, ingest order.

        Yields:
            One row per chunk, ordered by ``original_index`` (``doc_id``
            breaking ties) — the order a bundle's ``chunks.jsonl`` and its
            embedding matrix share.
        """
        for row in self._stream("varagity_snapshot_chunks", _CORPUS_CHUNKS_SQL):
            yield CorpusChunkRow(*row)

    def _stream(self, name: str, sql: str) -> Iterator[tuple[Any, ...]]:
        """Run one query through a server-side cursor inside the snapshot.

        A named (server-side) cursor fetches in batches, so a corpus of any
        size streams in bounded memory; binary results hand the vectors over
        bit-exact.

        Args:
            name: The server-side cursor's name.
            sql: The query to stream.

        Yields:
            The raw result rows, in query order.
        """
        with self._conn.cursor(name=name, binary=True) as cursor:
            cursor.itersize = _SNAPSHOT_BATCH_ROWS
            cursor.execute(sql)
            yield from cursor


def default_conninfo() -> str:
    """Build the connection string from settings.

    Shared by every Postgres-backed store (this one, the conversation
    store) and the migration runner, so ``POSTGRES_*`` is interpreted in
    exactly one place.

    Returns:
        A libpq conninfo string for the configured PostgreSQL instance.
    """
    settings = get_settings()
    return psycopg.conninfo.make_conninfo(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
    )


class ContextualVectorDB(ClosingContextMixin):
    """Dense vector store over PostgreSQL + pgvector.

    Owns one connection (autocommit; per-document writes group into an
    explicit transaction via :meth:`store_document`). Use as a context
    manager or call :meth:`close` when done.
    """

    def __init__(self, conninfo: str | None = None) -> None:
        """Connect and register the pgvector type adapters.

        Args:
            conninfo: libpq connection string; defaults to the
                ``POSTGRES_*`` settings.

        Raises:
            psycopg.OperationalError: If the database is unreachable.
        """
        self._conn = psycopg.connect(conninfo or default_conninfo(), autocommit=True)
        register_vector(self._conn)

    def close(self) -> None:
        """Close the underlying connection (idempotent)."""
        if not self._conn.closed:
            self._conn.close()

    def document_exists(self, doc_id: str, content_hash: str) -> bool:
        """Check whether a document was already ingested (idempotency, spec §8.2).

        Args:
            doc_id: The document's stable id.
            content_hash: The document's content hash (``doc_id`` already
                encodes it — checking both is belt-and-braces).

        Returns:
            ``True`` if a matching ``documents`` row exists.
        """
        row = self._conn.execute(
            "SELECT 1 FROM documents WHERE doc_id = %s AND content_hash = %s",
            (doc_id, content_hash),
        ).fetchone()
        return row is not None

    def document_n_chunks(self, doc_id: str) -> int | None:
        """Fetch a document's recorded chunk count.

        Lets the loader re-warn about known-empty documents (``n_chunks = 0``)
        without re-parsing them.

        Args:
            doc_id: The document's stable id.

        Returns:
            The ``n_chunks`` value, or ``None`` if the document is unknown.
        """
        row = self._conn.execute(
            "SELECT n_chunks FROM documents WHERE doc_id = %s", (doc_id,)
        ).fetchone()
        return None if row is None else int(row[0])

    def document_count(self) -> int:
        """Count the ingested documents.

        Backs the settings route's corpus-stale check (spec_v2 §4.7): an
        empty corpus has nothing to go stale, so a reingest-affecting
        override change only raises the flag when this is positive.

        Returns:
            Number of ``documents`` rows.
        """
        row = self._conn.execute("SELECT count(*) FROM documents").fetchone()
        return 0 if row is None else int(row[0])

    def chunk_count(self) -> int:
        """Count the stored chunks.

        Backs the ``varagity_corpus_chunks`` gauge (spec_v3 §6.1a): the
        store is the truthful source for "how big is my corpus", unlike the
        per-process ingest counters that reset with the API.

        Returns:
            Number of ``chunks`` rows.
        """
        row = self._conn.execute("SELECT count(*) FROM chunks").fetchone()
        return 0 if row is None else int(row[0])

    def document_count_by_type(self) -> dict[str, int]:
        """Count the ingested documents per file type.

        Backs the ``varagity_corpus_documents_by_type`` gauge (spec_v3
        §6.1a) — the store-derived replacement for the Ingestion
        dashboard's ``increase()``-over-a-flat-counter panel.

        Returns:
            ``file_type`` → document count; empty on an empty corpus.
        """
        rows = self._conn.execute(_DOCUMENTS_BY_TYPE_SQL).fetchall()
        return {str(file_type): int(count) for file_type, count in rows}

    def chunk_count_by_strategy(self) -> dict[str, int]:
        """Count the stored chunks per chunking strategy.

        Backs the ``varagity_corpus_chunks_by_strategy`` gauge (spec_v3
        §6.1a). The strategy is read from each chunk's ``metadata`` JSONB
        (it is not a column); chunks lacking the field count as
        ``unknown``.

        Returns:
            ``chunking_strategy`` → chunk count; empty on an empty corpus.
        """
        rows = self._conn.execute(_CHUNKS_BY_STRATEGY_SQL).fetchall()
        return {str(strategy): int(count) for strategy, count in rows}

    def list_documents(self) -> list[DocumentInfo]:
        """List every ingested document with its extraction mix.

        Backs ``GET /api/documents`` (spec_v2 §4.2): the corpus-management
        table of file, type, chunk count, ingested-at, and how many chunks
        came through OCR versus the digital text layer.

        Returns:
            One :class:`~varagity.stores.records.DocumentInfo` per
            ``documents`` row, newest ingest first (0-chunk documents
            included, with an empty mix).
        """
        rows = self._conn.execute(_LIST_DOCUMENTS_SQL).fetchall()
        return [_row_to_document(row) for row in rows]

    @contextmanager
    def iter_corpus(self) -> Iterator[CorpusSnapshot]:
        """Open one consistent snapshot of the whole corpus (a backup export).

        Everything a RAG bundle holds — every ``documents`` row, then every
        ``chunks`` row with its vector — is read inside **one** REPEATABLE
        READ, READ ONLY transaction (ADR-019), so the export is a single
        moment of the corpus even while an ingest or delete commits beside
        it. The rows stream through server-side cursors, so memory stays
        flat however large the corpus is. Readers never block writers in
        Postgres: the open snapshot holds no lock an ingest would wait on.

        Yields:
            The snapshot, valid until the ``with`` block exits (which ends
            the transaction).
        """
        with self._conn.transaction():
            self._conn.execute(_SNAPSHOT_ISOLATION_SQL)
            row = self._conn.execute(_CORPUS_COUNTS_SQL).fetchone()
            documents, chunks = (0, 0) if row is None else (int(row[0]), int(row[1]))
            yield CorpusSnapshot(self._conn, document_count=documents, chunk_count=chunks)

    @contextmanager
    def corpus_lock(self, *, exclusive: bool) -> Iterator[None]:
        """Hold the RAG corpus's advisory lock for the ``with`` block.

        Ingests hold it **shared** — they overlap each other exactly as they
        always have — and a backup restore holds it **exclusive**, so a
        restore and an ingest in *any* two processes exclude each other
        (ADR-019). The lock belongs to this store's database session: it is
        released when the block exits, and with the connection if the
        process dies first. Never blocks — a held lock is refused at once.

        Args:
            exclusive: Take it exclusively (a restore) rather than shared
                (an ingest).

        Yields:
            Nothing; the corpus lock is held until the block exits.

        Raises:
            CorpusLocked: If the lock is held in a mode that excludes this
                request.
        """
        row = self._conn.execute(_TRY_LOCK_SQL[exclusive], (CORPUS_LOCK_KEY,)).fetchone()
        if row is None or not row[0]:
            raise CorpusLocked(exclusive=exclusive)
        try:
            yield
        finally:
            try:
                self._conn.execute(_UNLOCK_SQL[exclusive], (CORPUS_LOCK_KEY,))
            except psycopg.Error as error:
                # The session is gone (or going): its end releases the lock.
                logger.warning("could not release the corpus lock (%s)", error)

    def corpus_lock_free(self) -> bool:
        """Report whether a restore could take the corpus lock right now.

        Takes the lock exclusively and gives it straight back — so ``False``
        means an ingest, in this process or another, holds it shared. The
        restore route's pre-202 check; the restore itself still takes the
        lock for real (and fails cleanly if an ingest won the race).

        Returns:
            ``True`` when nothing holds the corpus lock.
        """
        row = self._conn.execute(_TRY_LOCK_SQL[True], (CORPUS_LOCK_KEY,)).fetchone()
        if row is None or not row[0]:
            return False
        self._conn.execute(_UNLOCK_SQL[True], (CORPUS_LOCK_KEY,))
        return True

    def replace_corpus(
        self,
        documents: Iterable[CorpusDocumentRow],
        chunks: Iterable[CorpusChunkRow],
        *,
        on_indexing: Callable[[], None] | None = None,
    ) -> tuple[int, int]:
        """Replace every ``documents`` and ``chunks`` row in one transaction.

        The backup restore's write (ADR-019), in this order: empty both
        tables, drop the HNSW index (its definition saved from
        ``pg_indexes``), binary-``COPY`` the documents and then the chunks,
        rebuild the index from that definition, ``ANALYZE``, commit. The
        commit is where the restore takes effect: any failure before it —
        a row the bundle got wrong, a stream that turns out damaged, the
        caller raising mid-iteration — rolls the whole thing back and the
        previous corpus is untouched. ``TRUNCATE`` holds both tables until
        then, so a concurrent reader waits for the restored rows rather
        than seeing half of them.

        The two streams are consumed strictly in order (documents first, as
        the foreign key demands), so a chunk stream may depend on what the
        document stream has already yielded.

        Args:
            documents: Every document row, in any order.
            chunks: Every chunk row with its vector; each ``doc_id`` must be
                among ``documents``.
            on_indexing: Called once the rows are loaded, just before the
                vector index build — the slow tail of a large restore.

        Returns:
            ``(documents, chunks)`` written.

        Raises:
            CorpusTablesBusy: If another session held the tables past the
                lock timeout (nothing was replaced).
        """
        with self._conn.transaction():
            self._conn.execute(_RESTORE_LOCK_TIMEOUT_SQL)
            try:
                self._conn.execute(_TRUNCATE_CORPUS_SQL)
            except psycopg.errors.LockNotAvailable as error:
                raise CorpusTablesBusy(
                    "another database session kept the documents/chunks tables open — "
                    "nothing was replaced; try again"
                ) from error
            row = self._conn.execute(_INDEX_DEFINITION_SQL).fetchone()
            index_sql = _DEFAULT_HNSW_SQL if row is None else str(row[0])
            if row is not None:
                self._conn.execute(_DROP_HNSW_SQL)
            documents_written = self._copy_rows(
                _COPY_DOCUMENTS_SQL,
                _DOCUMENT_COPY_TYPES,
                (
                    (
                        document.doc_id,
                        document.source,
                        document.file_type,
                        document.content_hash,
                        document.n_chunks,
                        document.ingested_at,
                    )
                    for document in documents
                ),
            )
            chunks_written = self._copy_rows(
                _COPY_CHUNKS_SQL,
                _CHUNK_COPY_TYPES,
                (
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.original_index,
                        chunk.chunk_index,
                        chunk.content,
                        chunk.context,
                        chunk.contextualized_content,
                        chunk.embedding,
                        chunk.metadata,
                        chunk.created_at,
                    )
                    for chunk in chunks
                ),
            )
            if on_indexing is not None:
                on_indexing()
            self._conn.execute(index_sql)
            for statement in _ANALYZE_SQL:
                self._conn.execute(statement)
        logger.info(
            "replaced the corpus: %d document(s), %d chunk(s); %s rebuilt",
            documents_written,
            chunks_written,
            _HNSW_INDEX_NAME,
        )
        return documents_written, chunks_written

    def _copy_rows(self, sql: str, types: list[str], rows: Iterable[tuple[Any, ...]]) -> int:
        """Stream rows into one binary ``COPY … FROM STDIN``.

        Args:
            sql: The ``COPY`` statement.
            types: The Postgres type of each column, in order (binary COPY
                carries no type information of its own).
            rows: The column values, row by row.

        Returns:
            The rows written.
        """
        written = 0
        with self._conn.cursor() as cursor, cursor.copy(sql) as copy:
            copy.set_types(types)
            for values in rows:
                copy.write_row(values)
                written += 1
        return written

    def iter_index_rows(self) -> Generator[CorpusIndexRow, None, None]:
        """Stream what the BM25 index mirrors of every chunk.

        Backs the restore's BM25 rebuild (ADR-019): the five fields of each
        stored chunk, through a server-side cursor so the corpus streams in
        bounded memory. The transaction stays open until the iterator is
        exhausted or closed — close it (``contextlib.closing``) when
        abandoning it early.

        Yields:
            One row per chunk, ``original_index`` order.
        """
        with (
            self._conn.transaction(),
            self._conn.cursor(name="varagity_index_rows") as cursor,
        ):
            cursor.itersize = _SNAPSHOT_BATCH_ROWS
            cursor.execute(_INDEX_ROWS_SQL)
            for row in cursor:
                yield CorpusIndexRow(*row)

    def get_document(self, doc_id: str) -> DocumentInfo | None:
        """Fetch one ingested document by id.

        Backs the preview routes (ADR-010): resolving ``doc_id`` →
        ``source``/``content_hash`` happens once per locate/render request,
        so it gets a single-row lookup instead of a full-corpus list.

        Args:
            doc_id: The document's stable id.

        Returns:
            The document row with its extraction mix, or ``None`` when the
            id is unknown.
        """
        row = self._conn.execute(_GET_DOCUMENT_SQL, {"doc_id": doc_id}).fetchone()
        return None if row is None else _row_to_document(row)

    def delete_document(self, doc_id: str) -> int:
        """Delete a document row and, via FK cascade, all its chunks.

        Backs ``ingest --reingest``: pipeline-setting changes (e.g. toggling
        ``CONTEXTUALIZE``) don't change content hashes, so re-processing an
        unchanged file requires deleting its previous ingest first.

        Args:
            doc_id: The document's stable id (deleting an unknown id is a
                no-op).

        Returns:
            The number of ``documents`` rows deleted (0 or 1).
        """
        return self.delete_documents([doc_id])

    def delete_documents(self, doc_ids: Sequence[str]) -> int:
        """Delete several document rows and, via FK cascade, their chunks.

        Backs the corpus table's multi-select delete (spec_v2 §4.2): one
        statement for the whole set, so the marker rows the delete route
        writes last all fall in the same round trip.

        Args:
            doc_ids: The documents' stable ids (unknown ids are no-ops; an
                empty sequence skips the statement entirely).

        Returns:
            The number of ``documents`` rows deleted.
        """
        if not doc_ids:
            return 0
        cursor = self._conn.execute(
            "DELETE FROM documents WHERE doc_id = ANY(%s)", (list(doc_ids),)
        )
        return cursor.rowcount

    def next_original_index(self) -> int:
        """Allocate the next global chunk index (called once per ingest run).

        Returns:
            ``max(original_index) + 1`` across the corpus, or ``0`` when
            empty; the loader then increments monotonically in-process.
        """
        row = self._conn.execute(
            "SELECT COALESCE(MAX(original_index), -1) + 1 FROM chunks"
        ).fetchone()
        return 0 if row is None else int(row[0])

    def upsert_document(
        self, *, doc_id: str, source: str, file_type: str, content_hash: str, n_chunks: int
    ) -> None:
        """Insert or update one ``documents`` row.

        Args:
            doc_id: The document's stable id.
            source: Absolute file path (provenance).
            file_type: ``pdf`` / ``txt`` / ``md``.
            content_hash: The document's content hash.
            n_chunks: Number of chunks ingested (``0`` records a document
                with no extractable text).
        """
        self._conn.execute(
            _UPSERT_DOCUMENT_SQL,
            {
                "doc_id": doc_id,
                "source": source,
                "file_type": file_type,
                "content_hash": content_hash,
                "n_chunks": n_chunks,
            },
        )

    def upsert_chunks(self, records: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        """Insert or update one ``chunks`` row per record.

        Args:
            records: The chunk metadata records.
            embeddings: One vector per record (same order).

        Raises:
            ValueError: If ``records`` and ``embeddings`` lengths differ.
            psycopg.errors.UniqueViolation: If an insert collides on the
                ``(doc_id, original_index)`` identity (an ingest bug — the
                unique index exists to catch it early).
        """
        if len(records) != len(embeddings):
            raise ValueError(
                f"got {len(records)} records but {len(embeddings)} embeddings — "
                "every chunk needs exactly one vector"
            )
        params = [
            {
                "chunk_id": record.chunk_id,
                "doc_id": record.doc_id,
                "original_index": record.original_index,
                "chunk_index": record.chunk_index,
                "content": record.content,
                "context": record.context,
                "contextualized_content": record.contextualized_content,
                "embedding": Vector(embedding),
                "metadata": Json(record.model_dump(mode="json")),
                "created_at": record.created_at,
            }
            for record, embedding in zip(records, embeddings, strict=True)
        ]
        with self._conn.cursor() as cur:
            cur.executemany(_UPSERT_CHUNK_SQL, params)

    def store_document(
        self,
        *,
        doc_id: str,
        source: str,
        file_type: str,
        content_hash: str,
        records: list[ChunkRecord],
        embeddings: list[list[float]],
    ) -> None:
        """Atomically upsert a document row and all its chunks.

        One transaction per document: a partial failure rolls back the
        ``documents`` row too, so a re-run's idempotency check re-attempts
        the file instead of skipping a half-ingested one.

        Args:
            doc_id: The document's stable id.
            source: Absolute file path (provenance).
            file_type: ``pdf`` / ``txt`` / ``md``.
            content_hash: The document's content hash.
            records: The chunk metadata records.
            embeddings: One vector per record (same order).
        """
        with self._conn.transaction():
            self.upsert_document(
                doc_id=doc_id,
                source=source,
                file_type=file_type,
                content_hash=content_hash,
                n_chunks=len(records),
            )
            self.upsert_chunks(records, embeddings)

    def search(
        self, query_vector: list[float], k: int, verbose: int | None = None
    ) -> list[RetrievedChunk]:
        """Cosine top-k search (spec §11.2).

        Args:
            query_vector: The query embedding (e5 **query mode** — see
                :meth:`varagity.models.embeddings.EmbeddingsClient.embed_query`).
            k: Number of chunks to return.
            verbose: Console verbosity (0–2); defaults to
                ``settings.DEFAULT_VERBOSE``.

        Returns:
            The top-k chunks by cosine similarity, best first, with
            ``score = 1 - cosine_distance``.

        Raises:
            ValueError: If ``verbose`` is invalid.
        """
        check_verbose(get_settings().DEFAULT_VERBOSE if verbose is None else verbose)
        rows = self._conn.execute(_SEARCH_SQL, {"qvec": Vector(query_vector), "k": k}).fetchall()
        return [_row_to_retrieved(row) for row in rows]

    def document_chunks(self, doc_id: str) -> list[RetrievedChunk]:
        """Fetch one document's chunks in reading order.

        Backs the eval harness's fact-anchored golden resolution (spec_v2
        §7.4): a chunker sweep must locate which strategy-true chunk holds a
        planted fact, so it scans the document's stored contents rather than
        trusting index-anchored refs authored against other boundaries.

        Args:
            doc_id: The parent document id.

        Returns:
            The document's chunks ordered by ``chunk_index`` (empty when the
            document is unknown or has no extractable text). Chunks carry
            ``score = 0.0`` — there is no query here.
        """
        rows = self._conn.execute(_DOCUMENT_CHUNKS_SQL, {"doc_id": doc_id}).fetchall()
        return [_row_to_retrieved((*row, 0.0)) for row in rows]

    def fetch_by_identity(
        self, keys: list[tuple[str, int]]
    ) -> dict[tuple[str, int], RetrievedChunk]:
        """Fetch full chunk rows by their fusion identity (spec §11.4 hydrate).

        Backs the bm25/hybrid retrievers: Elasticsearch only stores the
        identity and text fields, so complete records (source, page, context,
        full metadata) are hydrated from here. The returned chunks carry
        ``score = 0.0`` — relevance belongs to the caller, which attaches its
        own (BM25 or fused) score.

        Args:
            keys: ``(doc_id, original_index)`` identity tuples.

        Returns:
            A mapping of identity tuple → chunk for every key that exists;
            unknown keys are simply absent (the caller decides whether that
            is an inconsistency worth logging).
        """
        if not keys:
            return {}
        rows = self._conn.execute(
            _FETCH_BY_IDENTITY_SQL,
            {
                "doc_ids": [doc_id for doc_id, _ in keys],
                "original_indexes": [original_index for _, original_index in keys],
            },
        ).fetchall()
        found: dict[tuple[str, int], RetrievedChunk] = {}
        for row in rows:
            chunk = _row_to_retrieved((*row, 0.0))
            found[(chunk.doc_id, chunk.original_index)] = chunk
        return found


def _row_to_document(row: tuple[Any, ...]) -> DocumentInfo:
    """Map a documents-table result row onto :class:`DocumentInfo`.

    Args:
        row: ``(doc_id, source, file_type, content_hash, n_chunks,
            ingested_at, extraction_mix)`` as selected by the document SQL.

    Returns:
        The typed document record.
    """
    doc_id, source, file_type, content_hash, n_chunks, ingested_at, mix = row
    return DocumentInfo(
        doc_id=doc_id,
        source=source,
        file_type=file_type,
        content_hash=content_hash,
        n_chunks=n_chunks,
        ingested_at=ingested_at,
        extraction_mix={name: int(count) for name, count in mix.items()},
    )


def _row_to_retrieved(row: tuple[Any, ...]) -> RetrievedChunk:
    """Map a search result row onto :class:`RetrievedChunk`.

    Args:
        row: ``(chunk_id, doc_id, original_index, content, context, metadata,
            score)`` as selected by the search SQL.

    Returns:
        The typed retrieval result.
    """
    chunk_id, doc_id, original_index, content, context, metadata, score = row
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        original_index=original_index,
        content=content,
        context=context,
        metadata=metadata,
        score=float(score),
    )
