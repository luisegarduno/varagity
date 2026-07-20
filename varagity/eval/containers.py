"""Ephemeral backing stores via testcontainers (plan decision #4).

The eval harness measures against throwaway Postgres/Elasticsearch
containers so the live corpus is never touched; the integration and e2e
test suites spin the same containers. This module is the single home for
that setup — image tags, schema application, and the single-node
Elasticsearch tuning — shared by both consumers.

``testcontainers`` lives in the ``eval`` dependency group (installed with
the dev group, or via ``uv run --group eval``); its import is deferred to
call time so importing this module — which the CLI does on every start via
the eval wiring — never requires it.

Operational notes baked in:

- The Elasticsearch container disables the disk-watermark allocation
  checks: on a host whose disk is >90% full, the default *percentage*
  watermarks refuse to allocate the throwaway index's primary shard and
  every operation times out. Ephemeral stores must never depend on host
  disk pressure.
- Single-node Elasticsearch is ``yellow`` by design; nothing here waits
  for ``green``.
"""

import logging
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

import psycopg

from varagity.stores.bm25_store import ElasticsearchBM25
from varagity.stores.vector_store import ContextualVectorDB

logger = logging.getLogger(__name__)

# Same images as docker-compose.yml / the schema the postgres container
# runs on first boot, so ephemeral behavior matches production.
POSTGRES_IMAGE = "pgvector/pgvector:pg16"
ES_IMAGE = "docker.elastic.co/elasticsearch/elasticsearch:9.2.0"
SCHEMA_PATH = Path(__file__).parents[1] / "stores" / "schema.sql"


@contextmanager
def ephemeral_postgres() -> Iterator[str]:
    """Run a throwaway pgvector Postgres with ``schema.sql`` applied.

    Yields:
        A libpq conninfo string for the container.

    Raises:
        ModuleNotFoundError: If ``testcontainers`` is not installed (the
            ``eval`` dependency group).
    """
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer(POSTGRES_IMAGE) as container:
        conninfo = psycopg.conninfo.make_conninfo(
            host=container.get_container_host_ip(),
            port=int(container.get_exposed_port(5432)),
            dbname=container.dbname,
            user=container.username,
            password=container.password,
        )
        with psycopg.connect(conninfo, autocommit=True) as conn:
            conn.execute(SCHEMA_PATH.read_text())
        logger.info("ephemeral pgvector Postgres up (schema applied)")
        yield conninfo


@contextmanager
def ephemeral_elasticsearch() -> Iterator[str]:
    """Run a throwaway single-node Elasticsearch (BM25 store backing).

    Yields:
        The container's base URL.

    Raises:
        ModuleNotFoundError: If ``testcontainers`` is not installed (the
            ``eval`` dependency group).
    """
    from testcontainers.elasticsearch import ElasticSearchContainer

    container = (
        ElasticSearchContainer(ES_IMAGE, mem_limit="2g")
        .with_env("discovery.type", "single-node")
        .with_env("ES_JAVA_OPTS", "-Xms512m -Xmx512m")
        # See the module docstring: never depend on host disk pressure.
        .with_env("cluster.routing.allocation.disk.threshold_enabled", "false")
    )
    with container as es:
        url = f"http://{es.get_container_host_ip()}:{es.get_exposed_port(9200)}"
        logger.info("ephemeral Elasticsearch up at %s", url)
        yield url


@dataclass(frozen=True)
class EphemeralStores:
    """Connected store clients over the throwaway containers.

    Attributes:
        store: Vector store on the ephemeral Postgres.
        bm25: BM25 store on the ephemeral Elasticsearch (index not yet
            created — the ingest loader creates it idempotently).
        pg_conninfo: The Postgres conninfo, for raw assertions.
        es_url: The Elasticsearch base URL, for raw assertions.
    """

    store: ContextualVectorDB
    bm25: ElasticsearchBM25
    pg_conninfo: str
    es_url: str


@contextmanager
def ephemeral_stores(index_name: str = "varagity_eval_bm25") -> Iterator[EphemeralStores]:
    """Run both throwaway containers and yield connected store clients.

    Args:
        index_name: BM25 index name inside the ephemeral Elasticsearch.

    Yields:
        The connected stores (closed, with their containers stopped, on
        exit).

    Raises:
        ModuleNotFoundError: If ``testcontainers`` is not installed (the
            ``eval`` dependency group).
    """
    with ExitStack() as stack:
        conninfo = stack.enter_context(ephemeral_postgres())
        es_url = stack.enter_context(ephemeral_elasticsearch())
        store = stack.enter_context(ContextualVectorDB(conninfo))
        bm25 = stack.enter_context(ElasticsearchBM25(url=es_url, index_name=index_name))
        yield EphemeralStores(store=store, bm25=bm25, pg_conninfo=conninfo, es_url=es_url)
