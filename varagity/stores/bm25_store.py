"""``ElasticsearchBM25`` — the sparse (contextual BM25) store.

Ported near-verbatim from the Anthropic cookbook's ``ElasticsearchBM25``
(spec §11.3): an index with the spec §8.3 mapping — ``content`` and
``contextualized_content`` analyzed with the built-in ``english`` analyzer
under BM25 similarity, identity fields stored but not indexed — plus bulk
indexing and a ``multi_match`` search over both text fields. Because chunks
are contextualized before they reach this store,
the BM25 index is *contextual* from its first document.

Identity fields are mapped ``"index": false`` but keep doc values, so
term-level queries against them (``delete_documents``' ``delete_by_query``)
still work — just via a slower doc-values scan, which is fine at dev scale.
"""

import logging
from collections.abc import Sequence
from types import TracebackType
from typing import Any

from elastic_transport import ConnectionError as ESConnectionError
from elastic_transport import ConnectionTimeout
from elasticsearch import ApiError, Elasticsearch, helpers
from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from varagity.config import get_settings
from varagity.debug.show import check_verbose
from varagity.stores.records import ChunkRecord

logger = logging.getLogger(__name__)

# Spec §8.3: analyzed text fields under the english analyzer + BM25
# similarity; identity fields stored but not indexed.
_INDEX_SETTINGS: dict[str, Any] = {
    "analysis": {"analyzer": {"default": {"type": "english"}}},
    "similarity": {"default": {"type": "BM25"}},
}
_INDEX_MAPPINGS: dict[str, Any] = {
    "properties": {
        "content": {"type": "text", "analyzer": "english"},
        "contextualized_content": {"type": "text", "analyzer": "english"},
        "doc_id": {"type": "keyword", "index": False},
        "chunk_id": {"type": "keyword", "index": False},
        "original_index": {"type": "integer", "index": False},
    }
}


def _is_transient(exc: BaseException) -> bool:
    """Classify an Elasticsearch failure as retryable or permanent.

    Args:
        exc: The exception raised by the client.

    Returns:
        ``True`` for connection/timeout trouble, 429, and 5xx responses;
        ``False`` for everything else (4xx like mapping errors are permanent
        and surface immediately).
    """
    if isinstance(exc, (ESConnectionError, ConnectionTimeout)):
        return True
    return isinstance(exc, ApiError) and (exc.status_code == 429 or exc.status_code >= 500)


# Index creation and bulks on a cold single node routinely exceed the
# client's 10s default; a timeout mid-create still creates the index
# server-side, so a generous timeout beats a confusing timeout-and-retry.
_REQUEST_TIMEOUT_S = 30

# Same backoff posture as the embeddings client: tenacity owns retries
# (the SDK's built-in retry is disabled so behavior is single-layered).
_es_retry = retry(
    retry=retry_if_exception(_is_transient),
    wait=wait_exponential(multiplier=0.5, max=10),
    stop=stop_after_attempt(4),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


class BM25Hit(BaseModel):
    """One BM25 search result (the spec §11.3 return fields).

    Carries the fusion identity plus the indexed texts — not the full
    metadata record, which lives in pgvector; retrievers hydrate complete
    rows by ``(doc_id, original_index)``.

    Attributes:
        doc_id: Parent document id.
        original_index: Global chunk index (fusion identity).
        content: Original chunk text.
        contextualized_content: The situated text that was BM25-indexed.
        score: Elasticsearch BM25 relevance score (unbounded; higher is
            better).
    """

    doc_id: str
    original_index: int
    content: str
    contextualized_content: str
    score: float


class ElasticsearchBM25:
    """Sparse keyword store over an Elasticsearch BM25 index.

    Owns one client. Use as a context manager or call :meth:`close` when
    done. The index is created lazily via :meth:`create_index` (the loader
    calls it idempotently at ingest start).
    """

    def __init__(self, *, url: str | None = None, index_name: str | None = None) -> None:
        """Create the client (no I/O until the first operation).

        Args:
            url: Elasticsearch base URL; defaults to
                ``settings.ELASTICSEARCH_URL``.
            index_name: BM25 index name; defaults to
                ``settings.BM25_INDEX_NAME``.
        """
        settings = get_settings()
        self.index_name = index_name or settings.BM25_INDEX_NAME
        self._client = Elasticsearch(
            url or settings.ELASTICSEARCH_URL,
            max_retries=0,  # tenacity owns retries (see module docstring)
            request_timeout=_REQUEST_TIMEOUT_S,
        )

    def close(self) -> None:
        """Close the underlying client (idempotent)."""
        self._client.close()

    def __enter__(self) -> "ElasticsearchBM25":
        """Enter a context that closes the client on exit.

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
        """Close the client on context exit.

        Args:
            exc_type: Exception type, if the block raised.
            exc: Exception instance, if the block raised.
            tb: Traceback, if the block raised.
        """
        self.close()

    @_es_retry
    def create_index(self) -> bool:
        """Create the BM25 index if it doesn't exist (idempotent).

        Returns:
            ``True`` if the index was created, ``False`` if it already
            existed.

        Raises:
            elastic_transport.ConnectionError: If Elasticsearch is still
                unreachable after retries.
        """
        if self._client.indices.exists(index=self.index_name):
            return False
        self._client.indices.create(
            index=self.index_name, settings=_INDEX_SETTINGS, mappings=_INDEX_MAPPINGS
        )
        logger.info("created BM25 index %r", self.index_name)
        return True

    @_es_retry
    def index_chunks(self, records: list[ChunkRecord]) -> int:
        """Bulk-index chunks, then refresh so they are searchable.

        Documents are addressed by ``chunk_id``, so re-indexing the same
        chunk overwrites instead of duplicating — the sparse-side counterpart
        of the vector store's ``ON CONFLICT (chunk_id) DO UPDATE``.

        Args:
            records: The chunk metadata records to index.

        Returns:
            The number of successfully indexed documents.

        Raises:
            elasticsearch.helpers.BulkIndexError: If any document is
                rejected (permanent — not retried).
            elastic_transport.ConnectionError: If Elasticsearch is still
                unreachable after retries.
        """
        if not records:
            return 0
        actions = [
            {
                "_index": self.index_name,
                "_id": record.chunk_id,
                "_source": {
                    "content": record.content,
                    "contextualized_content": record.contextualized_content,
                    "doc_id": record.doc_id,
                    "chunk_id": record.chunk_id,
                    "original_index": record.original_index,
                },
            }
            for record in records
        ]
        success, _ = helpers.bulk(self._client, actions)
        self._client.indices.refresh(index=self.index_name)
        return success

    def delete_document(self, doc_id: str) -> int:
        """Delete all of a document's chunks (``--reingest`` consistency).

        Keeps re-ingestion consistent across both stores: the vector store
        deletes via FK cascade, this store via ``delete_by_query``.

        Args:
            doc_id: The document's stable id (deleting an unknown id is a
                no-op).

        Returns:
            The number of chunks deleted.

        Raises:
            elastic_transport.ConnectionError: If Elasticsearch is still
                unreachable after retries.
        """
        return self.delete_documents([doc_id])

    @_es_retry
    def delete_documents(self, doc_ids: Sequence[str]) -> int:
        """Delete every chunk of several documents in one pass (bulk GC).

        One ``terms`` ``delete_by_query`` rather than a query per document:
        the whole set costs a single round trip and a single forced index
        refresh, which is what makes the corpus table's multi-select delete
        (spec_v2 §4.2) cheap. The retry lives here, so the single-document
        wrapper inherits it exactly once.

        Args:
            doc_ids: The documents' stable ids (unknown ids are no-ops; an
                empty sequence skips the round trip entirely).

        Returns:
            The number of chunks deleted across all of them.

        Raises:
            elastic_transport.ConnectionError: If Elasticsearch is still
                unreachable after retries.
        """
        if not doc_ids:
            return 0
        response = self._client.delete_by_query(
            index=self.index_name,
            query={"terms": {"doc_id": list(doc_ids)}},
            refresh=True,
        )
        return int(response["deleted"])

    @_es_retry
    def search(self, query: str, k: int, verbose: int | None = None) -> list[BM25Hit]:
        """BM25 top-k search via ``multi_match`` over both text fields.

        Args:
            query: The user's query, unformatted (Elasticsearch's ``english``
                analyzer does its own normalization).
            k: Number of hits to return.
            verbose: Console verbosity (0–2); defaults to
                ``settings.DEFAULT_VERBOSE``.

        Returns:
            The top-k hits, best first.

        Raises:
            ValueError: If ``verbose`` is invalid.
            elastic_transport.ConnectionError: If Elasticsearch is still
                unreachable after retries.
        """
        check_verbose(get_settings().DEFAULT_VERBOSE if verbose is None else verbose)
        response = self._client.search(
            index=self.index_name,
            query={
                "multi_match": {
                    "query": query,
                    "fields": ["content", "contextualized_content"],
                }
            },
            size=k,
        )
        return [
            BM25Hit(
                doc_id=hit["_source"]["doc_id"],
                original_index=hit["_source"]["original_index"],
                content=hit["_source"]["content"],
                contextualized_content=hit["_source"]["contextualized_content"],
                score=hit["_score"],
            )
            for hit in response["hits"]["hits"]
        ]
