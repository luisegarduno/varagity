"""Prefect ingestion flow: the spec §9 pipeline as tracked task runs.

The flow delegates to :func:`varagity.ingest.loader.ingest_corpus` — the
single orchestration loop — passing an
:class:`~varagity.ingest.loader.IngestStages` bundle whose entries are
``@task``-wrapped equivalents of the loader's stage functions, so every
stage of every file becomes a task run visible at the Prefect UI (``:4200``)
with state, duration, and logs.

Model/store tasks (contextualize, embed, store) carry ``retries=2`` with
exponential backoff. This is a second retry layer on purpose: the
``tenacity`` retries *inside* the clients cover transient HTTP failures
within one call, while task retries re-run the whole stage after the client
has given up — e.g. an Elasticsearch restart mid-ingest (both store writes
are idempotent, so a re-run is safe). Discovery/parse/chunk are local and
deterministic; retrying them cannot help, so they carry none.

Result caching is disabled on every task (``NO_CACHE``): the stages are
side-effecting calls against live services, and their inputs include
unhashable handles (store/client objects, the ``rich`` progress display)
that make Prefect's default input-hash cache policy log an error per run —
a cache hit could never be correct here anyway.
"""

from pathlib import Path

from langchain_core.documents import Document
from prefect import flow, task
from prefect.cache_policies import NO_CACHE
from prefect.logging import get_run_logger
from prefect.tasks import exponential_backoff
from rich.progress import Progress

from varagity.ingest import loader
from varagity.ingest.discovery import Buckets
from varagity.ingest.loader import IngestStages, IngestSummary, ingest_corpus
from varagity.ingest.parsers import Parser, RawDocument
from varagity.models import EmbeddingsClient, LLMClient
from varagity.stores import ChunkRecord, ContextualVectorDB, ElasticsearchBM25


@task(name="discover_documents", cache_policy=NO_CACHE)
def discover_documents_task(docs_path: str, verbose: int | None = None) -> Buckets:
    """Task wrapper over corpus discovery (spec §9.1).

    Args:
        docs_path: Directory to scan.
        verbose: Console verbosity (0–2); defaults to
            ``settings.DEFAULT_VERBOSE``.

    Returns:
        The bucketed file paths.
    """
    buckets = loader.discover_documents(docs_path, verbose=verbose)
    get_run_logger().info(
        "discovered %d file(s) under %s (%d text-like, %d pdf)",
        buckets.total,
        docs_path,
        len(buckets.text_like),
        len(buckets.pdf),
    )
    return buckets


@task(name="parse_document", cache_policy=NO_CACHE)
def parse_document_task(parser: Parser, path: Path, *, verbose: int) -> RawDocument:
    """Task wrapper over text extraction (spec §9.2).

    Args:
        parser: Parser for the file's discovery bucket.
        path: The file to parse.
        verbose: Validated console verbosity.

    Returns:
        The extracted document.
    """
    raw = loader.parse_document(parser, path, verbose=verbose)
    get_run_logger().info(
        "parsed %s → %d chars (extraction=%s)",
        path.name,
        len(raw.text),
        raw.source_meta.get("extraction", "text"),
    )
    return raw


@task(name="chunk_document", cache_policy=NO_CACHE)
def chunk_document_task(raw: RawDocument, *, verbose: int) -> list[Document]:
    """Task wrapper over chunking (spec §9.3).

    Args:
        raw: The parsed document.
        verbose: Validated console verbosity.

    Returns:
        The chunks, provenance seeded into each chunk's metadata.
    """
    chunks = loader.chunk_document(raw, verbose=verbose)
    get_run_logger().info(
        "chunked %s into %d chunk(s)", raw.source_meta.get("file_name"), len(chunks)
    )
    return chunks


@task(
    name="contextualize_chunks",
    retries=2,
    retry_delay_seconds=exponential_backoff(backoff_factor=2),
    cache_policy=NO_CACHE,
)
def contextualize_chunks_task(
    *,
    document_text: str,
    chunk_texts: list[str],
    llm: LLMClient | None,
    file_name: str,
    progress: Progress,
    verbose: int,
) -> list[str | None]:
    """Task wrapper over situating-blurb generation (spec §9.4).

    Args:
        document_text: The parent document's full extracted text.
        chunk_texts: The document's chunk texts, in order.
        llm: Chat client, or ``None`` for the identity path
            (``settings.CONTEXTUALIZE`` off).
        file_name: Source file name (labels the sub-progress bar).
        progress: The run's progress display.
        verbose: Validated console verbosity.

    Returns:
        One context blurb per chunk (all ``None`` when ``llm`` is ``None``).
    """
    contexts = loader.contextualize_chunks(
        document_text=document_text,
        chunk_texts=chunk_texts,
        llm=llm,
        file_name=file_name,
        progress=progress,
        verbose=verbose,
    )
    logger = get_run_logger()
    if llm is None:
        logger.info("CONTEXTUALIZE off — identity path for %s", file_name)
    else:
        logger.info("contextualized %d chunk(s) of %s", len(contexts), file_name)
    return contexts


@task(
    name="embed_chunks",
    retries=2,
    retry_delay_seconds=exponential_backoff(backoff_factor=2),
    cache_policy=NO_CACHE,
)
def embed_chunks_task(
    texts: list[str], *, embeddings: EmbeddingsClient, verbose: int
) -> list[list[float]]:
    """Task wrapper over passage embedding (spec §9.5).

    Args:
        texts: The ``contextualized_content`` of each chunk, in order.
        embeddings: The embeddings client.
        verbose: Validated console verbosity.

    Returns:
        One embedding vector per text, in order.
    """
    vectors = loader.embed_chunks(texts, embeddings=embeddings, verbose=verbose)
    get_run_logger().info("embedded %d passage(s)", len(vectors))
    return vectors


@task(
    name="store_chunks",
    retries=2,
    retry_delay_seconds=exponential_backoff(backoff_factor=2),
    cache_policy=NO_CACHE,
)
def store_chunks_task(
    records: list[ChunkRecord],
    vectors: list[list[float]],
    *,
    store: ContextualVectorDB,
    bm25: ElasticsearchBM25,
) -> None:
    """Task wrapper over the dual-store write (spec §9.6).

    Both writes share this one task boundary: a partial failure fails the
    task loudly and the retry re-runs both writes (idempotent — see
    :func:`varagity.ingest.loader.store_chunks` for the ordering rationale).

    Args:
        records: The document's chunk records (non-empty).
        vectors: One embedding per record, in order.
        store: The vector store.
        bm25: The BM25 store.
    """
    loader.store_chunks(records, vectors, store=store, bm25=bm25)
    get_run_logger().info(
        "stored %d chunk(s) of %s in both stores", len(records), records[0].file_name
    )


# The task-wrapped seam threaded through the loader's single run loop.
_TASK_STAGES = IngestStages(
    discover=discover_documents_task,
    parse=parse_document_task,
    chunk=chunk_document_task,
    contextualize=contextualize_chunks_task,
    embed=embed_chunks_task,
    store=store_chunks_task,
)


@flow(name="ingest", validate_parameters=False)
def ingest_flow(
    docs_path: str | None = None,
    *,
    store: ContextualVectorDB | None = None,
    bm25: ElasticsearchBM25 | None = None,
    embeddings: EmbeddingsClient | None = None,
    llm: LLMClient | None = None,
    reingest: bool = False,
    verbose: int | None = None,
) -> IngestSummary:
    """Ingest the corpus with every stage tracked as a Prefect task run.

    A thin ``@flow`` shell over :func:`varagity.ingest.loader.ingest_corpus`
    (same parameters, same behavior — one file failing is counted, logged,
    and visible as a failed task run, never aborting the corpus).

    Parameter validation is off because callers (tests, the Phase 9 eval
    harness) inject duck-typed store/client fakes that pydantic would
    reject; the flow's inputs are already-validated internals.

    Args:
        docs_path: Corpus directory; defaults to ``settings.DOCS_PATH``.
        store: Vector store to write to; constructed from settings (and
            closed on return) when omitted.
        bm25: BM25 store to write to; constructed from settings (and closed
            on return) when omitted.
        embeddings: Embeddings client; resolved via the model registry when
            omitted.
        llm: Chat client for contextualization; resolved via the model
            registry when omitted. Unused when ``settings.CONTEXTUALIZE``
            is off.
        reingest: Delete each discovered document's previous ingest (from
            both stores) and re-process it.
        verbose: Console verbosity (0–2); defaults to
            ``settings.DEFAULT_VERBOSE``.

    Returns:
        The run's counters.
    """
    return ingest_corpus(
        docs_path,
        store=store,
        bm25=bm25,
        embeddings=embeddings,
        llm=llm,
        reingest=reingest,
        verbose=verbose,
        stages=_TASK_STAGES,
    )
