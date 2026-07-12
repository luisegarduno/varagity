"""Persistent stores: pgvector, Elasticsearch BM25, and the data model.

Chunks live in **both** stores, joinable by ``(doc_id, original_index)``
(spec §8). ``schema.sql`` in this directory is the database DDL, mounted
into the postgres container's ``/docker-entrypoint-initdb.d/`` (first-boot
only); the ``migrations/`` directory holds the v2 additive DDL applied by
the idempotent runner (:mod:`varagity.stores.migrate`) on API startup.
Conversation history (spec_v2 §9.1) shares the Postgres instance via
:class:`~varagity.stores.conversation_store.ConversationStore`.
"""

from varagity.stores.bm25_store import BM25Hit, ElasticsearchBM25
from varagity.stores.conversation_store import ConversationStore
from varagity.stores.migrate import run_migrations
from varagity.stores.records import ChunkRecord, RetrievedChunk, content_hash, derive_doc_id
from varagity.stores.vector_store import ContextualVectorDB

__all__ = [
    "BM25Hit",
    "ChunkRecord",
    "ContextualVectorDB",
    "ConversationStore",
    "ElasticsearchBM25",
    "RetrievedChunk",
    "content_hash",
    "derive_doc_id",
    "run_migrations",
]
