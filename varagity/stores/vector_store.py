"""``ContextualVectorDB`` — the pgvector-backed dense vector store.

The Anthropic cookbook's in-memory ``ContextualVectorDB`` responsibilities
(store vectors + metadata, cosine ``search``), re-backed with
PostgreSQL/pgvector (spec §11.2) so the index is durable, concurrent, and
inspectable via SQL. The schema itself lives in ``schema.sql`` (applied by
the postgres container on first boot).
"""

import logging
from types import TracebackType
from typing import Any

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.types.json import Json

from varagity.config import get_settings
from varagity.debug.show import check_verbose
from varagity.stores.records import ChunkRecord, RetrievedChunk

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


def _default_conninfo() -> str:
    """Build the connection string from settings.

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


class ContextualVectorDB:
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
        self._conn = psycopg.connect(conninfo or _default_conninfo(), autocommit=True)
        register_vector(self._conn)

    def close(self) -> None:
        """Close the underlying connection (idempotent)."""
        if not self._conn.closed:
            self._conn.close()

    def __enter__(self) -> "ContextualVectorDB":
        """Enter a context that closes the connection on exit.

        Returns:
            This store.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the connection on context exit.

        Args:
            exc_type: Exception type, if the block raised.
            exc: Exception instance, if the block raised.
            tb: Traceback, if the block raised.
        """
        self.close()

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
        cursor = self._conn.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
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
