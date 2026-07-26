"""The graph subsystem — a second, graph-based retrieval corpus (spec_graphrag §5.2, §10).

GraphRAG is a **peer** subsystem beside chunk-RAG, not a sixth retriever: it
ingests structured messages (an iMessage ``chat.db`` in v1) into a knowledge
graph with per-message provenance. Stage 1 lands the engine-independent
foundations — the message-source registry family and the iMessage parser —
plus the ``bake-off`` infrastructure that decides the engine (ADR-017); the
engine, graph build pipeline, query path, and web surfaces are stage 2.

This package re-exports the message-source surface for convenience; the family
lives under :mod:`varagity.graph.sources`.
"""

from varagity.graph.sources import (
    MESSAGE_SOURCE_REGISTRY,
    MessageBatch,
    MessageSource,
    SourceMessage,
    Tapback,
    batch_for_path,
    find_message_source,
    get_message_source,
    register,
)

__all__ = [
    "MESSAGE_SOURCE_REGISTRY",
    "MessageBatch",
    "MessageSource",
    "SourceMessage",
    "Tapback",
    "batch_for_path",
    "find_message_source",
    "get_message_source",
    "register",
]
