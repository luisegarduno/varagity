"""Graph-engine protocol and registry (spec_graphrag §5.2, §8; the spec §5.1 pattern).

Each engine module defines one adapter decorated with ``@register("name")``;
callers resolve one by name from :attr:`~varagity.config.Settings.GRAPH_ENGINE`
via :func:`get_graph_engine`. ADR-017 chose ``lightrag`` and the losing
bake-off adapters were deleted at stage-2 start; **the seam is what stays**,
so the decision remains benchmark-revisitable exactly as the retriever and
chat-engine registries are.

The family is split in two on purpose:

* :class:`GraphEngine` is the registered singleton — stateless, zero-arg
  constructible, and cheap to import (adapters import their heavy engine libs
  *inside* :meth:`GraphEngine.session`, stage-1 decision #8).
* :class:`GraphSession` holds every piece of per-run state — the engine
  handle, its working directory, its event loop, the transcript→guid index —
  behind a context manager, because teardown is not optional.

:meth:`GraphSession.build` takes a **sequence of batches and upserts at
message grain**: adapters guid-merge across source files
(:func:`varagity.graph.render.merge_batches`) before rendering, so an
overlapping ``chat.db`` upload or a re-export of a grown database never
duplicates a message. A second ``build`` call with an overlapping batch must
not duplicate either, and — the harder half — a *changed* document must not
be silently kept: the shipped adapter diffs rendered content hashes against
:mod:`varagity.graph.manifest` and deletes before re-inserting.

The rest of the protocol is what an application needs beyond a benchmark
(stage-2 decision #11): :meth:`GraphSession.resume` to finish a killed build,
:meth:`GraphSession.retrieve` to feed a *streamed* answer the app writes
itself, :meth:`GraphSession.export` to draw the graph,
:meth:`GraphSession.delete_documents` to retract sources, and
:meth:`GraphSession.document_statuses` to report progress. Sessions are
**single-writer per working directory** by engine invariant, which is why
exactly one process (the API) ever opens one.
"""

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from varagity.graph.records import (
    BuildReport,
    GraphAnswer,
    GraphExport,
    GraphRetrieval,
    GraphStats,
)
from varagity.graph.sources.base import MessageBatch


class GraphSession(Protocol):
    """One engine's open working session over one working directory.

    Sessions are handed out by :meth:`GraphEngine.session` and are valid only
    inside its ``with`` block.
    """

    def build(
        self,
        batches: Sequence[MessageBatch],
        *,
        verbose: int = 0,
        prune_removed: bool = True,
    ) -> BuildReport:
        """Index (or re-index) messages into the graph — an upsert, not an append.

        Re-callable by design: a build killed halfway through resumes by
        being called again, and one over an unchanged corpus costs no
        extraction at all.

        Args:
            batches: Parsed source files. Messages are guid-merged across
                them before rendering, so overlapping exports of the same
                conversation collapse to one message stream.
            verbose: Validated console verbosity (0–2).
            prune_removed: Whether ``batches`` render the **whole** corpus,
                making a document the graph holds but the render omits a
                deleted source. A bounded build (message cap, date floor)
                passes ``False``: its render is partial on purpose.

        Returns:
            What the build did: messages seen, wall clock, and any failures
            the adapter caught and continued past.
        """
        ...

    def resume(self, *, verbose: int = 0) -> BuildReport:
        """Finish whatever a killed build left in flight, without re-rendering.

        Args:
            verbose: Validated console verbosity (0–2).

        Returns:
            What the pass did. ``messages_seen`` describes the corpus already
            indexed, not a fresh hand-off.
        """
        ...

    def retrieve(
        self, question: str, *, mode: str | None = None, verbose: int = 0
    ) -> GraphRetrieval:
        """Find the evidence for one question, writing no answer at all.

        The shipped query path's first half (ADR-017's retrieval-only
        decision): the app streams its own grounded answer over what this
        returns, so an engine must be able to retrieve *without* spending its
        own answer call. :meth:`query` is this composed with an answer stage,
        which is what keeps the harness measuring the shipped diet.

        Args:
            question: The search query, verbatim — already condensed by the
                chat engine when the turn needed it (the answer prompt still
                gets the user's own words; spec_v3 §4.2).
            mode: Engine query mode; ``None`` uses the adapter's primary.
            verbose: Validated console verbosity (0–2).

        Returns:
            The evidence and transcript passages, empty when the engine
            could not retrieve (a graph turn degrades to "no facts", it does
            not raise).
        """
        ...

    def query(self, question: str, *, mode: str | None = None, verbose: int = 0) -> GraphAnswer:
        """Answer one question from the graph.

        Args:
            question: The user's question, verbatim (golden ``kind`` metadata
                is never shown to an engine — stage-1 decision #10).
            mode: Engine query mode; ``None`` uses the adapter's primary mode
                (stage-1 decision #13 — the ``--mode`` escape hatch records
                extra passes).
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

    def document_statuses(self) -> dict[str, int]:
        """Count the engine's documents by processing status.

        Returns:
            Status name → document count, in the engine's own vocabulary, or
            an empty mapping when it would not say (which a progress reader
            must treat as "no news", never as "zero documents").
        """
        ...

    def export(
        self,
        label: str = "*",
        *,
        max_depth: int = 3,
        max_nodes: int = 1000,
    ) -> GraphExport:
        """Read a renderable slice of the graph.

        Args:
            label: Entity name to centre the slice on; ``"*"`` takes the whole
                graph, ordered so the cap keeps the most connected nodes.
            max_depth: Hops to walk out from ``label``.
            max_nodes: Node cap; the engine may clamp it further and says so
                through :attr:`~varagity.graph.records.GraphExport.truncated`.

        Returns:
            The slice, empty when the graph cannot be read.
        """
        ...

    def delete_documents(self, doc_keys: Sequence[str]) -> int:
        """Remove documents, and whatever the graph derived from them.

        Args:
            doc_keys: Transcript document keys to retract.

        Returns:
            How many deletions the engine accepted.
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
