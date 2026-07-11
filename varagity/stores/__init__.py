"""Persistent stores: pgvector, Elasticsearch BM25, and the data model.

Chunks live in **both** stores, joinable by ``(doc_id, original_index)``
(spec §8). ``schema.sql`` in this directory is the database DDL, mounted
into the postgres container's ``/docker-entrypoint-initdb.d/`` (first-boot
only).
"""

from varagity.stores.bm25_store import BM25Hit, ElasticsearchBM25
from varagity.stores.records import ChunkRecord, RetrievedChunk, content_hash, derive_doc_id
from varagity.stores.vector_store import ContextualVectorDB

__all__ = [
    "BM25Hit",
    "ChunkRecord",
    "ContextualVectorDB",
    "ElasticsearchBM25",
    "RetrievedChunk",
    "content_hash",
    "derive_doc_id",
]
