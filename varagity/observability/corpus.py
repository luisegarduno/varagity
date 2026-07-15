"""Store-derived corpus gauges: the spec_v3 §6.1a scrape-time collector.

The ingest counters in :mod:`varagity.observability.metrics` answer "what
did *this process* do" — and they are the wrong instrument for "how big is
my corpus". Ingestion is rare and bursty, so a counter sits flat for hours
and ``increase()`` over it is ``0``; worse, after an API restart the series
is absent and then *reappears at its full value*, so Prometheus never sees
the ``0 → 1`` rise and ``increase()`` is ``0`` even over ``$__range``. That
is what made every Ingestion panel read zero while the metrics themselves
were correct (spec_v3 §2.3, §6.1).

Gauges read from pgvector at scrape time have neither problem: they reflect
store state, survive an API restart, and — unlike the counters (ADR-007) —
they also show CLI ingests, which record into a never-scraped registry.

Failure posture: a store outage must degrade ``/metrics``, never 500 it.
The last good snapshot is re-served (going stale) and a fresh process with
no snapshot yet emits the gauges with no samples at all.
:data:`varagity.observability.metrics.DEPENDENCY_UP` is what reports store
health; these gauges deliberately do not.
"""

import logging
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

import psycopg
from prometheus_client import REGISTRY
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector, CollectorRegistry

from varagity.stores.vector_store import ContextualVectorDB

logger = logging.getLogger(__name__)

# Prometheus scrapes every 15 s (observability/prometheus/prometheus.yml).
# A TTL just under that keeps one scrape to one round of queries while
# still collapsing the extra collect() calls a manual curl or a second
# scraper adds.
CACHE_TTL_SECONDS = 10.0

# name → label names, for the metric catalog the dashboard guard test
# checks panel expressions against (spec_v3 §6.4).
CORPUS_GAUGES: dict[str, tuple[str, ...]] = {
    "varagity_corpus_documents": (),
    "varagity_corpus_chunks": (),
    "varagity_corpus_documents_by_type": ("file_type",),
    "varagity_corpus_chunks_by_strategy": ("chunking_strategy",),
}

_DOCS = "varagity_corpus_documents"
_CHUNKS = "varagity_corpus_chunks"
_DOCS_BY_TYPE = "varagity_corpus_documents_by_type"
_CHUNKS_BY_STRATEGY = "varagity_corpus_chunks_by_strategy"


@dataclass(frozen=True)
class CorpusSnapshot:
    """One scrape's worth of corpus counts.

    Attributes:
        documents: Total ingested documents.
        chunks: Total stored chunks.
        documents_by_type: ``file_type`` → document count.
        chunks_by_strategy: ``chunking_strategy`` → chunk count.
    """

    documents: int = 0
    chunks: int = 0
    documents_by_type: dict[str, int] = field(default_factory=dict)
    chunks_by_strategy: dict[str, int] = field(default_factory=dict)


class CorpusCollector(Collector):
    """Expose the corpus gauges, querying pgvector at scrape time.

    Registered in the default registry by the API's app factory when
    ``METRICS_ENABLED`` (:func:`register_corpus_collector`). A short TTL
    (:data:`CACHE_TTL_SECONDS`) bounds the query rate; a store outage
    serves the last good snapshot rather than raising.
    """

    def __init__(
        self,
        store_factory: Callable[[], ContextualVectorDB] = ContextualVectorDB,
        ttl_seconds: float = CACHE_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Configure the collector's store access and caching.

        Args:
            store_factory: Builds a connected store per refresh (override
                seam for tests). Called inside a ``with`` block, so the
                connection closes after every refresh.
            ttl_seconds: How long a snapshot is served before re-querying.
            clock: Monotonic time source (override seam for tests).
        """
        self._store_factory = store_factory
        self._ttl = ttl_seconds
        self._clock = clock
        self._snapshot: CorpusSnapshot | None = None
        self._fetched_at: float | None = None

    def describe(self) -> Iterable[GaugeMetricFamily]:
        """Declare the gauge families without touching the store.

        Registration calls this to detect duplicate names; without it
        ``prometheus_client`` would call :meth:`collect` at registration
        time, turning app startup into a database query.

        Returns:
            The empty gauge families this collector emits.
        """
        return self._families(CorpusSnapshot())

    def collect(self) -> Iterator[GaugeMetricFamily]:
        """Emit the corpus gauges for one scrape.

        Yields:
            The gauge families, populated from a fresh snapshot, the
            cached one, or — if the store has never been reachable in this
            process — with no samples.
        """
        snapshot = self._current()
        if snapshot is None:
            yield from self._families(None)
            return
        yield from self._families(snapshot)

    def _current(self) -> CorpusSnapshot | None:
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
        except psycopg.Error as error:
            # Serve stale over raising: a scrape that 500s takes the whole
            # /metrics endpoint down, including the metrics that are fine.
            logger.warning(
                "corpus gauges not refreshed — postgres unreachable (%s); serving %s",
                error,
                "the last snapshot" if self._snapshot is not None else "no samples",
            )
        return self._snapshot

    def _fetch(self) -> CorpusSnapshot:
        """Query the store for the current corpus counts.

        Returns:
            A fresh snapshot.

        Raises:
            psycopg.Error: If the database is unreachable or the queries
                fail; the caller degrades to the cached snapshot.
        """
        with self._store_factory() as store:
            return CorpusSnapshot(
                documents=store.document_count(),
                chunks=store.chunk_count(),
                documents_by_type=store.document_count_by_type(),
                chunks_by_strategy=store.chunk_count_by_strategy(),
            )

    def _families(self, snapshot: CorpusSnapshot | None) -> list[GaugeMetricFamily]:
        """Render a snapshot as gauge families.

        Args:
            snapshot: The counts to render, or ``None`` to emit the
                families with no samples (never-reachable store).

        Returns:
            One family per :data:`CORPUS_GAUGES` entry, in catalog order.
        """
        documents = GaugeMetricFamily(_DOCS, "Documents currently in the corpus (from pgvector).")
        chunks = GaugeMetricFamily(_CHUNKS, "Chunks currently stored (from pgvector).")
        by_type = GaugeMetricFamily(
            _DOCS_BY_TYPE,
            "Documents currently in the corpus, by file type (from pgvector).",
            labels=["file_type"],
        )
        by_strategy = GaugeMetricFamily(
            _CHUNKS_BY_STRATEGY,
            "Chunks currently stored, by the strategy that produced them (from pgvector).",
            labels=["chunking_strategy"],
        )
        if snapshot is not None:
            documents.add_metric([], snapshot.documents)
            chunks.add_metric([], snapshot.chunks)
            for file_type, count in sorted(snapshot.documents_by_type.items()):
                by_type.add_metric([file_type], count)
            for strategy, count in sorted(snapshot.chunks_by_strategy.items()):
                by_strategy.add_metric([strategy], count)
        return [documents, chunks, by_type, by_strategy]


def register_corpus_collector(
    registry: CollectorRegistry = REGISTRY,
    collector: CorpusCollector | None = None,
) -> CorpusCollector:
    """Register the corpus collector, replacing any previous registration.

    Idempotent on purpose: ``create_app()`` runs once per process in
    production but many times across a test session, and re-registering the
    same gauge names in the process-wide default registry would raise.

    Args:
        registry: The registry to register into (defaults to the
            process-wide one that ``GET /metrics`` serves).
        collector: The collector to register; a default-configured one is
            built when omitted.

    Returns:
        The registered collector.
    """
    collector = collector or CorpusCollector()
    for existing in list(getattr(registry, "_collector_to_names", {})):
        if isinstance(existing, CorpusCollector):
            registry.unregister(existing)
    registry.register(collector)
    return collector
