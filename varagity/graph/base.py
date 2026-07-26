"""Graph-engine protocol and registry (spec_graphrag §5.2, §8; the spec §5.1 pattern).

Each engine module defines one adapter decorated with ``@register("name")``;
callers resolve one with :func:`get_graph_engine`. Stage 1 registers the three
ADR-017 bake-off candidates (``lightrag``, ``cognee``, ``graphiti``) and the
bake-off *enumerates* the registry rather than selecting from it — there is no
``GRAPH_ENGINE`` setting yet (plan decision #7: settings land with their first
consumer, which is stage 2's runtime query path). The losing adapters are
deleted after the ADR; **the seam is what stays**, so the decision remains
benchmark-revisitable exactly as the retriever and chat-engine registries are.

The family is split in two on purpose:

* :class:`GraphEngine` is the registered singleton — stateless, zero-arg
  constructible, and cheap to import (adapters import their heavy engine libs
  *inside* :meth:`GraphEngine.session`, plan decision #8).
* :class:`GraphSession` holds every piece of per-run state — the engine
  handle, its working directory, the transcript→guid index — behind a context
  manager, because teardown is not optional: Graphiti's embedded FalkorDB Lite
  spawns a local ``redis-server`` subprocess, and each adapter owns an asyncio
  event loop.

:meth:`GraphSession.build` takes a **sequence of batches and upserts at
message grain**: adapters guid-merge across source files
(:func:`varagity.graph.render.merge_batches`) before rendering, so an
overlapping ``chat.db`` upload or a re-export of a grown database never
duplicates a message. A second ``build`` call with an overlapping batch must
not duplicate either — each adapter maps that to its engine's own mechanism
(LightRAG doc-status ids, cognee content-hash update-in-place, Graphiti
episode identity).
"""

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from varagity.graph.records import BuildReport, GraphAnswer, GraphStats
from varagity.graph.sources.base import MessageBatch


class GraphSession(Protocol):
    """One engine's open working session over one working directory.

    Sessions are handed out by :meth:`GraphEngine.session` and are valid only
    inside its ``with`` block.
    """

    def build(self, batches: Sequence[MessageBatch], *, verbose: int = 0) -> BuildReport:
        """Index (or re-index) messages into the graph — an upsert, not an append.

        Args:
            batches: Parsed source files. Messages are guid-merged across
                them before rendering, so overlapping exports of the same
                conversation collapse to one message stream.
            verbose: Validated console verbosity (0–2).

        Returns:
            What the build did: messages seen, wall clock, and any failures
            the adapter caught and continued past.
        """
        ...

    def query(self, question: str, *, mode: str | None = None, verbose: int = 0) -> GraphAnswer:
        """Answer one question from the graph.

        Args:
            question: The user's question, verbatim (golden ``kind`` metadata
                is never shown to an engine — plan decision #10).
            mode: Engine query mode; ``None`` uses the adapter's primary mode
                (plan decision #13 — the ``--mode`` escape hatch records extra
                passes).
            verbose: Validated console verbosity (0–2).

        Returns:
            The answer with its graph evidence and the mode actually used.
        """
        ...

    def stats(self) -> GraphStats:
        """Report the current graph size, as far as the engine will say.

        Returns:
            Entity/relation/community counts, each ``None`` where the engine
            exposes no way to ask (the incremental-reindex check reads this
            before and after a delta build).
        """
        ...


@runtime_checkable
class GraphEngine(Protocol):
    """Opens working sessions against a graph engine (spec_graphrag §8).

    ``runtime_checkable`` for the same reason
    :class:`~varagity.chat.base.ChatEngine` is: the protocol will appear in
    stage-2 Prefect flow signatures, and Prefect builds a pydantic parameter
    schema from the annotations at decoration time, which requires types
    usable with ``isinstance``.
    """

    def session(self, workdir: Path) -> AbstractContextManager[GraphSession]:
        """Open a session whose state lives under ``workdir``.

        Engines self-store (plan decision #9): every byte an engine writes —
        graph, vectors, KV, caches — belongs under ``workdir``, so wiping it
        is a clean slate and no compose service is involved.

        Args:
            workdir: The engine's working directory (created if absent).

        Returns:
            A context manager yielding the session; exiting it tears down
            whatever the engine started (subprocesses, event loops, stores).
        """
        ...


GRAPH_ENGINE_REGISTRY: dict[str, GraphEngine] = {}


def register[T: type[Any]](name: str) -> Callable[[T], T]:
    """Class decorator registering a graph-engine instance under ``name``.

    Args:
        name: Registry key (the ``--engine`` value the harness takes).

    Returns:
        The decorator, which instantiates and registers the class unchanged.
    """

    def deco(cls: T) -> T:
        GRAPH_ENGINE_REGISTRY[name] = cls()
        return cls

    return deco


def get_graph_engine(name: str) -> GraphEngine:
    """Look up a registered graph engine by name.

    Args:
        name: Registry key (e.g. ``"lightrag"``).

    Returns:
        The registered engine instance.

    Raises:
        KeyError: If no engine is registered under ``name`` (the message
            lists the available ones).
    """
    if name not in GRAPH_ENGINE_REGISTRY:
        raise KeyError(f"Unknown graph engine {name!r}. Available: {list(GRAPH_ENGINE_REGISTRY)}")
    return GRAPH_ENGINE_REGISTRY[name]
