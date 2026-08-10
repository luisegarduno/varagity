"""Workdir-derived graph gauges: the ADR-013 shape, applied to the graph.

The graph corpus has exactly the problem the chunk-RAG corpus had before
ADR-013: "how big is my graph" is a **state** question, and an in-process
counter cannot answer it. A backfill runs for hours and then stops; the
counter that recorded it is born at its full value after the next API
restart, so ``increase()`` over it is ``0`` — over any window. Gauges read
at scrape time have neither problem.

Where they read from is the difference. The corpus gauges query pgvector;
these read the **files in the engine's working directory**, and never open
the engine (stage-2 decision #18):

* ``varagity_graph_documents{status}`` counts the engine's own doc-status
  records — the same pending/processing/processed/failed vocabulary the
  Graph RAG tab shows, so a stalled or failed backfill is visible on the
  dashboard rather than only in the tab.
* ``varagity_graph_entities`` / ``_relations`` / ``_messages`` come from the
  summary sidecar every build and delete refreshes
  (:mod:`varagity.graph.manifest`), which exists precisely so a status poll
  or a scrape never walks a multi-megabyte graphml.

Opening a session here would be worse than slow: the engine is
single-writer per working directory and this process is that writer, so a
scrape must not initialize storages a build is holding — and on an *unbuilt*
graph it would create the workdir, turning a Prometheus scrape into "the
graph now exists".

Failure posture matches the corpus collector's. An unreadable file serves
the last good snapshot (going stale) rather than raising, because a scrape
that 500s takes ``/metrics`` down including the metrics that are fine. An
**unbuilt** graph is not an error and not a zero: the families are emitted
with no samples at all, the same honesty
:class:`~varagity.api.schemas.GraphStatusOut` keeps when it reports ``None``
rather than ``0`` for a graph nothing has been extracted into yet.
"""

import json
import logging
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from prometheus_client import REGISTRY
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector, CollectorRegistry

from varagity.graph.manifest import load_summary
from varagity.graph.service import graph_workdir

logger = logging.getLogger(__name__)

# Matches the corpus collector's window (Prometheus scrapes every 15 s), so
# one scrape costs one read of each file and a manual curl beside it costs
# nothing extra.
CACHE_TTL_SECONDS = 10.0

# LightRAG's `JsonDocStatusStorage` file. Reading an engine's own storage
# file is a deliberate, cheap coupling: the alternative is opening the
# session, which is the one thing a scrape must not do. A different engine
# (or a renamed file) makes this read miss, which degrades to "no
# doc-status samples" — the other three gauges, and the graph, are
# unaffected.
DOC_STATUS_FILENAME = "kv_store_doc_status.json"

# name → label names, for the catalog the dashboard guard test checks panel
# expressions against (spec_v3 §6.4).
GRAPH_GAUGES: dict[str, tuple[str, ...]] = {
    "varagity_graph_documents": ("status",),
    "varagity_graph_entities": (),
    "varagity_graph_relations": (),
    "varagity_graph_messages": (),
}

_DOCUMENTS = "varagity_graph_documents"
_ENTITIES = "varagity_graph_entities"
_RELATIONS = "varagity_graph_relations"
_MESSAGES = "varagity_graph_messages"


@dataclass(frozen=True)
class GraphSnapshot:
    """One scrape's worth of graph size, as the workdir reports it.

    Every field is optional-shaped because "nothing has been built here"
    and "the graph would not say" are both real states, and a ``0`` would
    read like an empty graph on a dashboard.

    Attributes:
        documents_by_status: Engine doc-status name → document count.
        entities: Nodes in the graph, or ``None`` when unknown.
        relations: Edges in the graph, or ``None`` when unknown.
        messages: Distinct messages the manifest accounts for, or ``None``.
    """

    documents_by_status: dict[str, int] = field(default_factory=dict)
    entities: int | None = None
    relations: int | None = None
    messages: int | None = None


def read_document_statuses(workdir: Path) -> dict[str, int]:
    """Count the engine's documents by processing status, from disk.

    Args:
        workdir: The engine's working directory.

    Returns:
        Status name → document count. An absent file is an empty mapping:
        a workdir that has never had a document enqueued genuinely has no
        statuses, which is not a failure.

    Raises:
        OSError: If the file exists but cannot be read.
        ValueError: If it is not readable JSON — the caller degrades to the
            last good snapshot rather than reporting a suddenly empty graph.
    """
    path = workdir / DOC_STATUS_FILENAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path} is not a document-status mapping")
    counts: Counter[str] = Counter()
    for record in raw.values():
        status = record.get("status") if isinstance(record, Mapping) else None
        counts[str(status) if status else "unknown"] += 1
    return dict(counts)


class GraphCollector(Collector):
    """Expose the graph gauges, reading the workdir at scrape time.

    Registered in the default registry by the API's app factory when
    ``METRICS_ENABLED`` (:func:`register_graph_collector`). A short TTL
    (:data:`CACHE_TTL_SECONDS`) bounds the read rate; an unreadable workdir
    serves the last good snapshot rather than raising.
    """

    def __init__(
        self,
        workdir_factory: Callable[[], Path] = graph_workdir,
        ttl_seconds: float = CACHE_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Configure the collector's workdir access and caching.

        Args:
            workdir_factory: Resolves the graph's working directory per
                refresh (override seam for tests). Deliberately the shared
                settings derivation rather than the graph service's own
                property: a scrape must not depend on the engine registry
                being warm, nor construct the service singleton. Called per
                refresh, not once, so a settings change lands without a
                restart.
            ttl_seconds: How long a snapshot is served before re-reading.
            clock: Monotonic time source (override seam for tests).
        """
        self._workdir_factory = workdir_factory
        self._ttl = ttl_seconds
        self._clock = clock
        self._snapshot: GraphSnapshot | None = None
        self._fetched_at: float | None = None

    def describe(self) -> Iterable[GaugeMetricFamily]:
        """Declare the gauge families without touching the workdir.

        Registration calls this to detect duplicate names; without it
        ``prometheus_client`` would call :meth:`collect` at registration
        time, turning app startup into a disk read.

        Returns:
            The empty gauge families this collector emits.
        """
        return self._families(None)

    def collect(self) -> Iterator[GaugeMetricFamily]:
        """Emit the graph gauges for one scrape.

        Yields:
            The gauge families, populated from a fresh snapshot, the cached
            one, or — for an unbuilt graph or a workdir that has never been
            readable — with no samples.
        """
        yield from self._families(self._current())

    def _current(self) -> GraphSnapshot | None:
        """Return a snapshot, refreshing it when the TTL has expired.

        Returns:
            The fresh snapshot, the still-valid cached one, the stale one
            when a refresh fails, or ``None`` when no refresh has ever
            succeeded.
        """
        now = self._clock()
        if (
            self._snapshot is not None
            and self._fetched_at is not None
            and now - self._fetched_at < self._ttl
        ):
            return self._snapshot
        try:
            self._snapshot = self._fetch()
            self._fetched_at = now
        except (OSError, ValueError) as error:
            # Serve stale over raising: /metrics must survive a workdir a
            # build is churning through, and the corpus gauges beside these
            # are none of this failure's business.
            logger.warning(
                "graph gauges not refreshed — the workdir would not read (%s); serving %s",
                error,
                "the last snapshot" if self._snapshot is not None else "no samples",
            )
        return self._snapshot

    def _fetch(self) -> GraphSnapshot:
        """Read the workdir's sidecars for the current graph size.

        Returns:
            A fresh snapshot; an empty one when the working directory does
            not exist, which renders as no samples rather than zeros.

        Raises:
            OSError: If a present file cannot be read.
            ValueError: If the doc-status file is not readable JSON.
        """
        workdir = self._workdir_factory()
        if not workdir.is_dir():
            return GraphSnapshot()
        summary = load_summary(workdir)
        return GraphSnapshot(
            documents_by_status=read_document_statuses(workdir),
            entities=summary.entities if summary is not None else None,
            relations=summary.relations if summary is not None else None,
            messages=summary.message_guids if summary is not None else None,
        )

    def _families(self, snapshot: GraphSnapshot | None) -> list[GaugeMetricFamily]:
        """Render a snapshot as gauge families.

        Args:
            snapshot: The counts to render, or ``None`` to emit the
                families with no samples (never-readable workdir).

        Returns:
            One family per :data:`GRAPH_GAUGES` entry, in catalog order.
            Fields the workdir does not know stay sample-less: an unbuilt
            graph must not report itself as a graph with zero entities.
        """
        documents = GaugeMetricFamily(
            _DOCUMENTS,
            "Transcript documents the graph engine holds, by processing status.",
            labels=["status"],
        )
        entities = GaugeMetricFamily(_ENTITIES, "Entities in the extracted graph.")
        relations = GaugeMetricFamily(_RELATIONS, "Relations in the extracted graph.")
        messages = GaugeMetricFamily(
            _MESSAGES, "Messages the graph's workdir manifest accounts for."
        )
        if snapshot is not None:
            for status, count in sorted(snapshot.documents_by_status.items()):
                documents.add_metric([status], count)
            if snapshot.entities is not None:
                entities.add_metric([], snapshot.entities)
            if snapshot.relations is not None:
                relations.add_metric([], snapshot.relations)
            if snapshot.messages is not None:
                messages.add_metric([], snapshot.messages)
        return [documents, entities, relations, messages]


def register_graph_collector(
    registry: CollectorRegistry = REGISTRY,
    collector: GraphCollector | None = None,
) -> GraphCollector:
    """Register the graph collector, replacing any previous registration.

    Idempotent for the corpus collector's reason: ``create_app()`` runs once
    per process in production but many times across a test session, and
    re-registering the same gauge names in the process-wide default registry
    would raise.

    Args:
        registry: The registry to register into (defaults to the
            process-wide one that ``GET /metrics`` serves).
        collector: The collector to register; a default-configured one is
            built when omitted.

    Returns:
        The registered collector.
    """
    collector = collector or GraphCollector()
    for existing in list(getattr(registry, "_collector_to_names", {})):
        if isinstance(existing, GraphCollector):
            registry.unregister(existing)
    registry.register(collector)
    return collector
