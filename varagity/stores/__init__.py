"""Persistent stores: the pgvector-backed vector store and its data model.

``schema.sql`` in this directory is the database DDL, mounted into the
postgres container's ``/docker-entrypoint-initdb.d/`` (first-boot only).
"""

from varagity.stores.records import ChunkRecord, RetrievedChunk, content_hash, derive_doc_id
from varagity.stores.vector_store import ContextualVectorDB

__all__ = [
    "ChunkRecord",
    "ContextualVectorDB",
    "RetrievedChunk",
    "content_hash",
    "derive_doc_id",
]
