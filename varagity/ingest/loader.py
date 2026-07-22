r"""Ingestion orchestrator: parse → chunk → contextualize → embed → store (spec §9).

Contextualization (spec §9.4): when ``settings.CONTEXTUALIZE`` is
on, each chunk gets an LLM-generated situating blurb and
``contextualized_content = context + "\n\n" + content`` is what gets
embedded; when off, the identity path (``context = None``,
``contextualized_content = content``) is preserved — the non-contextual eval
baseline (plan decision #2). A document's chunks are contextualized
sequentially, in order, so llama.cpp can reuse its prompt cache across the
shared document preamble.

Dual-write (spec §9.6): every chunk lands in **both** stores —
Elasticsearch (contextual BM25) first, pgvector second — within the same
per-file boundary; a store failure after client-level retries fails that
file's ingest loudly. The ordering is deliberate: the pgvector ``documents``
row is the idempotency marker, so it must commit *last* — a failure between
the two writes leaves no marker, the next run re-attempts the file, and the
Elasticsearch bulk (addressed by deterministic ``chunk_id``) overwrites
rather than duplicates.

Idempotency: a file whose ``(doc_id, content_hash)`` is already recorded is
skipped *before* parsing. Pipeline-setting changes (``CONTEXTUALIZE``, chunk
params) don't change content hashes, so re-processing an unchanged corpus
requires ``reingest=True`` (the CLI's ``ingest --reingest``), which deletes
each discovered document from both stores before ingesting it fresh. A file
with no extractable text is never silently dropped: it gets a ``documents``
row with ``n_chunks = 0``, a warning, and a dedicated summary count.

Orchestration seam: each spec §9 stage is a named module function
(:func:`parse_document`, :func:`chunk_document`, :func:`contextualize_chunks`,
:func:`embed_chunks`, :func:`store_chunks`), and the run loop invokes them
through an :class:`IngestStages` bundle. By default the bundle holds the
functions themselves; ``varagity.pipeline.ingest_flow`` passes task-wrapped
equivalents so every stage becomes a tracked, retryable Prefect task run —
one orchestration loop, with or without Prefect.
"""

import logging
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from langchain_core.documents import Document
from rich.progress import Progress, TaskID

from varagity.chunking import get_chunker
from varagity.config import Settings, get_settings
from varagity.context import situate_context
from varagity.debug import show
from varagity.debug.show import check_verbose
from varagity.ingest.discovery import Buckets, discover_documents
from varagity.ingest.parsers import Parser, RawDocument, get_parser
from varagity.models import EmbeddingsClient, LLMClient, get_model
from varagity.stores import (
    ChunkRecord,
    ContextualVectorDB,
    ElasticsearchBM25,
    content_hash,
    derive_doc_id,
)

logger = logging.getLogger(__name__)

# Below this many non-whitespace characters a parse is treated as "no
# extractable text" (the reference notebook's trigger, plan decision #10).
MIN_EXTRACTED_CHARS = 50

# Discovery bucket attribute → parser registry name.
_BUCKET_PARSERS: tuple[tuple[str, str], ...] = (
    ("text_like", "text"),
    ("pdf", "pdf"),
    ("office", "office"),
    ("web", "web"),
)


@dataclass
class IngestSummary:
    """Counters for one ingest run, rendered by the CLI summary table.

    Attributes:
        discovered: Files found in the corpus buckets.
        ingested: Files parsed, chunked, embedded, and stored this run.
        skipped: Unchanged files skipped via the idempotency check.
        no_text: Files with no extractable text (recorded as 0-chunk
            documents; includes known-empty files seen again).
        unsupported: Files whose bucket has no registered parser (a
            defensive counter — every v1 bucket has one; it guards future
            buckets added to discovery before their parser lands).
        failed: Files that raised during ingestion (logged, run continues).
        chunks: Total chunks stored this run.
    """

    discovered: int = 0
    ingested: int = 0
    skipped: int = 0
    no_text: int = 0
    unsupported: int = 0
    failed: int = 0
    chunks: int = 0


def file_timestamps(path: Path) -> tuple[datetime | None, datetime | None]:
    """Read a file's filesystem timestamps — document provenance, not ingest time.

    These land on every chunk's metadata record (``file_created_at`` /
    ``file_modified_at``), answering "how old is this source?" independently
    of ``created_at`` (which records when the chunk was ingested).

    Birth time is best-effort by nature: ``os.stat`` exposes
    ``st_birthtime`` on macOS/Windows but not on Linux (CPython doesn't
    surface ``statx``), so :func:`_birthtime_fallback` shells out to GNU
    coreutils there; filesystems without birth times, or a copy/download
    that reset them, yield ``None`` rather than a guess (``st_ctime`` is
    inode-change time, not creation, and would lie).

    Args:
        path: The file to stat.

    Returns:
        A ``(created, modified)`` pair of aware UTC datetimes; either is
        ``None`` when unavailable (both, if the file cannot be stat'd).
    """
    try:
        stat = path.stat()
    except OSError:
        return None, None
    modified = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
    birth = getattr(stat, "st_birthtime", None) or _birthtime_fallback(path)
    created = None if birth is None else datetime.fromtimestamp(birth, tz=UTC)
    return created, modified


def _birthtime_fallback(path: Path) -> float | None:
    """Read a file's birth time via GNU coreutils (``stat -c %W``).

    The Linux leg of :func:`file_timestamps`: the kernel exposes birth
    times through ``statx()``, which CPython doesn't wrap — but coreutils'
    ``stat`` does. One short subprocess per *file* (not per chunk) is noise
    next to parsing and embedding.

    Args:
        path: The file to stat.

    Returns:
        The birth time as epoch seconds, or ``None`` when unknown (``%W``
        prints ``0``), on non-GNU ``stat`` implementations, or on any
        subprocess failure.
    """
    try:
        result = subprocess.run(
            ["stat", "-c", "%W", "--", str(path)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        birth = float(result.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return birth if birth > 0 else None


def parse_document(parser: Parser, path: Path, *, verbose: int) -> RawDocument:
    """Extract a file's text and provenance (spec §9.2).

    Args:
        parser: Parser for the file's discovery bucket.
        path: The file to parse.
        verbose: Validated console verbosity.

    Returns:
        The extracted document.
    """
    return parser.extract(path, verbose=verbose)


def chunk_document(raw: RawDocument, *, verbose: int) -> list[Document]:
    """Split a parsed document with the configured strategy (spec §9.3).

    Args:
        raw: The parsed document (text + provenance).
        verbose: Validated console verbosity.

    Returns:
        The chunks, provenance seeded into each chunk's metadata.

    Raises:
        KeyError: If ``settings.CHUNKING_STRATEGY`` names an unregistered
            strategy.
    """
    chunker = get_chunker(get_settings().CHUNKING_STRATEGY)
    return chunker.split(raw.text, source_meta=raw.source_meta, verbose=verbose)


def contextualize_chunks(
    *,
    document_text: str,
    chunk_texts: list[str],
    llm: LLMClient | None,
    file_name: str,
    progress: Progress,
    verbose: int,
) -> list[str | None]:
    """Generate one situating blurb per chunk, or the identity path (spec §9.4).

    Chunks are processed sequentially, in document order, so every call
    shares the same document preamble and llama.cpp reuses its prompt cache.
    Progress is a per-chunk sub-bar under the run's file bar.

    Args:
        document_text: The parent document's full extracted text.
        chunk_texts: The document's chunk texts, in order.
        llm: Chat client for contextualization, or ``None`` for the identity
            path (every context ``None`` — ``settings.CONTEXTUALIZE`` off).
        file_name: Source file name (labels the sub-progress bar).
        progress: The run's progress display.
        verbose: Validated console verbosity.

    Returns:
        One context blurb per chunk, in chunk order (all ``None`` when
        ``llm`` is ``None``).
    """
    if llm is None:
        return [None] * len(chunk_texts)
    contexts: list[str | None] = []
    sub_task: TaskID = progress.add_task(f"  ↳ contextualizing {file_name}", total=len(chunk_texts))
    try:
        for chunk_text in chunk_texts:
            contexts.append(situate_context(document_text, chunk_text, llm=llm, verbose=verbose))
            progress.advance(sub_task)
    finally:
        progress.remove_task(sub_task)
    return contexts


def embed_chunks(
    texts: list[str], *, embeddings: EmbeddingsClient, verbose: int
) -> list[list[float]]:
    """Embed contextualized chunk texts in e5 passage mode (spec §9.5).

    Args:
        texts: The ``contextualized_content`` of each chunk, in order.
        embeddings: The embeddings client.
        verbose: Validated console verbosity.

    Returns:
        One embedding vector per text, in order.

    Raises:
        openai.APIError: If embedding still fails after client retries.
    """
    return embeddings.embed_passages(texts, verbose=verbose)


def store_chunks(
    records: list[ChunkRecord],
    vectors: list[list[float]],
    *,
    store: ContextualVectorDB,
    bm25: ElasticsearchBM25,
) -> None:
    """Write one document's chunks to both stores (spec §9.6).

    Both writes share one boundary, BM25 first: the pgvector ``documents``
    row is the idempotency marker and must commit last, so a failure in
    between leaves the file re-attemptable on the next run (the
    deterministic ``chunk_id``-addressed BM25 docs then overwrite rather
    than duplicate). The parent-document row's fields are taken from the
    records, which all belong to one document.

    Args:
        records: The document's chunk records (non-empty; the empty-
            extraction guard runs before this stage).
        vectors: One embedding per record, in order.
        store: The vector store.
        bm25: The BM25 store.

    Raises:
        ValueError: If ``records`` is empty.
    """
    if not records:
        raise ValueError("store_chunks requires at least one record")
    head = records[0]
    bm25.index_chunks(records)
    store.store_document(
        doc_id=head.doc_id,
        source=head.source,
        file_type=head.file_type,
        content_hash=head.content_hash,
        records=records,
        embeddings=vectors,
    )


@dataclass(frozen=True)
class IngestStages:
    """Call seam through which the run loop invokes each spec §9 stage.

    Defaults are the plain stage functions in this module, so constructing
    the bundle without arguments changes nothing. ``varagity.pipeline``
    substitutes Prefect ``@task``-wrapped equivalents (same signatures), so
    the same loop yields tracked, retryable task runs — the orchestration
    logic exists once.

    Attributes:
        discover: Corpus discovery (spec §9.1).
        parse: Text extraction (spec §9.2).
        chunk: Chunking (spec §9.3).
        contextualize: Situating-blurb generation (spec §9.4).
        embed: Passage embedding (spec §9.5).
        store: Dual-store write (spec §9.6).
    """

    discover: Callable[..., Buckets] = field(default=discover_documents)
    parse: Callable[..., RawDocument] = field(default=parse_document)
    chunk: Callable[..., list[Document]] = field(default=chunk_document)
    contextualize: Callable[..., list[str | None]] = field(default=contextualize_chunks)
    embed: Callable[..., list[list[float]]] = field(default=embed_chunks)
    store: Callable[..., Any] = field(default=store_chunks)


def ingest_corpus(
    docs_path: str | None = None,
    *,
    store: ContextualVectorDB | None = None,
    bm25: ElasticsearchBM25 | None = None,
    embeddings: EmbeddingsClient | None = None,
    llm: LLMClient | None = None,
    reingest: bool = False,
    verbose: int | None = None,
    stages: IngestStages | None = None,
    on_file: Callable[[Path, str, int], None] | None = None,
) -> IngestSummary:
    """Ingest every supported document under ``docs_path`` into both stores.

    Args:
        docs_path: Corpus directory; defaults to ``settings.DOCS_PATH``.
        store: Vector store to write to; constructed from settings (and
            closed on return) when omitted.
        bm25: BM25 store to write to; constructed from settings (and closed
            on return) when omitted. Its index is created idempotently at
            run start.
        embeddings: Embeddings client; resolved via the model registry when
            omitted.
        llm: Chat client for contextualization; resolved via the model
            registry when omitted. Unused when ``settings.CONTEXTUALIZE``
            is off.
        reingest: Delete each discovered document's previous ingest (from
            both stores) and re-process it. Needed after pipeline-setting
            changes (``CONTEXTUALIZE``, chunk params): those don't change
            content hashes, so unchanged files are otherwise skipped.
        verbose: Console verbosity (0–2); defaults to
            ``settings.DEFAULT_VERBOSE``.
        stages: Per-stage call seam; the plain stage functions when omitted.
            ``varagity.pipeline.ingest_flow`` passes task-wrapped stages so
            each stage is a tracked Prefect task run.
        on_file: Observer called after each file with ``(path, outcome,
            chunks_stored)``, where outcome is ``"ingested"`` /
            ``"skipped"`` / ``"no_text"`` / ``"failed"`` — the seam the
            API's live ingest-progress stream rides (spec_v2 §4.2). A
            raising observer is logged and ignored: progress reporting
            must never fail a run.

    Returns:
        The run's counters (one file failing is counted and logged, not
        raised — a bad file must not abort a corpus run).

    Raises:
        ValueError: If ``verbose`` is invalid.
        psycopg.OperationalError: If the vector store is unreachable.
        elastic_transport.ConnectionError: If Elasticsearch is unreachable
            after retries (index creation at run start).
    """
    settings = get_settings()
    verbose = check_verbose(settings.DEFAULT_VERBOSE if verbose is None else verbose)
    root = Path(docs_path if docs_path is not None else settings.DOCS_PATH)
    stages = stages if stages is not None else IngestStages()

    buckets = stages.discover(str(root), verbose=verbose)
    summary = IngestSummary(discovered=buckets.total)

    # Resolve parsers up front; a bucket without one is
    # counted and skipped loudly, never silently dropped.
    work: list[tuple[Parser, Path]] = []
    for bucket_name, parser_name in _BUCKET_PARSERS:
        paths: list[Path] = getattr(buckets, bucket_name)
        if not paths:
            continue
        try:
            parser = get_parser(parser_name)
        except KeyError:
            logger.warning(
                "no parser registered for %r — skipping %d file(s)",
                parser_name,
                len(paths),
            )
            summary.unsupported += len(paths)
            continue
        work.extend((parser, path) for path in paths)

    owns_store = store is None
    active_store = store if store is not None else ContextualVectorDB()
    owns_bm25 = bm25 is None
    active_bm25 = bm25 if bm25 is not None else ElasticsearchBM25()
    try:
        active_bm25.create_index()
        client = embeddings if embeddings is not None else get_model("embedding")
        # The LLM is only needed (and only resolved) when contextualizing.
        llm_client: LLMClient | None = None
        if settings.CONTEXTUALIZE:
            llm_client = llm if llm is not None else get_model("default")
        next_index = active_store.next_original_index()
        with Progress(console=show.console, disable=verbose == 0) as progress:
            task = progress.add_task("Ingesting", total=len(work))
            for parser, path in work:
                progress.update(task, description=f"Ingesting {path.name}")
                try:
                    outcome, n_chunks = _ingest_file(
                        path=path,
                        root=root,
                        parser=parser,
                        store=active_store,
                        bm25=active_bm25,
                        embeddings=client,
                        llm=llm_client,
                        settings=settings,
                        next_index=next_index,
                        reingest=reingest,
                        progress=progress,
                        verbose=verbose,
                        stages=stages,
                    )
                except Exception:
                    logger.exception("failed to ingest %s — continuing with the next file", path)
                    summary.failed += 1
                    _notify_file(on_file, path, "failed", 0)
                else:
                    if outcome == "ingested":
                        summary.ingested += 1
                    elif outcome == "skipped":
                        summary.skipped += 1
                    else:
                        summary.no_text += 1
                    summary.chunks += n_chunks
                    next_index += n_chunks
                    _notify_file(on_file, path, outcome, n_chunks)
                progress.advance(task)
    finally:
        if owns_store:
            active_store.close()
        if owns_bm25:
            active_bm25.close()
    return summary


def _notify_file(
    on_file: Callable[[Path, str, int], None] | None, path: Path, outcome: str, n_chunks: int
) -> None:
    """Invoke the per-file observer, containing its failures.

    Args:
        on_file: The observer, or ``None`` (no-op).
        path: The file just finished.
        outcome: Its summary outcome.
        n_chunks: Chunks stored for it this run.
    """
    if on_file is None:
        return
    try:
        on_file(path, outcome, n_chunks)
    except Exception:  # a progress observer must never fail the run
        logger.warning("on_file observer raised for %s", path, exc_info=True)


def _ingest_file(
    *,
    path: Path,
    root: Path,
    parser: Parser,
    store: ContextualVectorDB,
    bm25: ElasticsearchBM25,
    embeddings: EmbeddingsClient,
    llm: LLMClient | None,
    settings: Settings,
    next_index: int,
    reingest: bool,
    progress: Progress,
    verbose: int,
    stages: IngestStages,
) -> tuple[Literal["ingested", "skipped", "no_text"], int]:
    """Ingest a single file into both stores.

    Args:
        path: The file to ingest.
        root: The corpus root (``doc_id`` hashes the path relative to it,
            plan decision #6).
        parser: Parser for the file's bucket.
        store: The vector store.
        bm25: The BM25 store (written before the vector store — see the
            module docstring for the ordering rationale).
        embeddings: The embeddings client.
        llm: Chat client for contextualization, or ``None`` to keep the
            identity path (``context = None``,
            ``contextualized_content = content``).
        settings: Loaded application settings.
        next_index: First free global ``original_index`` for this file.
        reingest: Delete the document's previous ingest (if any) from both
            stores instead of skipping it as unchanged.
        progress: The run's progress display (hosts the per-chunk
            contextualization sub-bar).
        verbose: Validated console verbosity.
        stages: Per-stage call seam (plain functions or Prefect tasks).

    Returns:
        A ``(summary_field, chunks_stored)`` pair, where ``summary_field``
        is one of ``"ingested"``, ``"skipped"``, or ``"no_text"``.
    """
    file_hash = content_hash(path.read_bytes())
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:  # symlink target outside the corpus root
        relative = path.name
    doc_id = derive_doc_id(relative, file_hash)
    source = str(path.resolve())
    file_type = path.suffix.lower().lstrip(".")

    if reingest:
        # BM25 first: if it fails, the pgvector documents row (the
        # idempotency marker) is still intact and the next run retries both.
        deleted_bm25 = bm25.delete_document(doc_id)
        if store.delete_document(doc_id) or deleted_bm25:
            logger.info("%s: --reingest — deleted previous ingest, re-processing", relative)
    elif store.document_exists(doc_id, file_hash):
        if store.document_n_chunks(doc_id) == 0:
            logger.warning(
                "%s: known document with no extractable text (unchanged since last run)", relative
            )
            return "no_text", 0
        logger.info("%s: unchanged — skipping (already ingested)", relative)
        return "skipped", 0

    raw_doc = stages.parse(parser, path, verbose=verbose)
    if sum(1 for char in raw_doc.text if not char.isspace()) < MIN_EXTRACTED_CHARS:
        logger.warning(
            "%s: no extractable text (<%d non-whitespace chars) — recording a 0-chunk document",
            relative,
            MIN_EXTRACTED_CHARS,
        )
        store.upsert_document(
            doc_id=doc_id, source=source, file_type=file_type, content_hash=file_hash, n_chunks=0
        )
        return "no_text", 0

    file_created_at, file_modified_at = file_timestamps(path)
    chunks = stages.chunk(raw_doc, verbose=verbose)
    contexts = stages.contextualize(
        document_text=raw_doc.text,
        chunk_texts=[chunk.page_content for chunk in chunks],
        llm=llm,
        file_name=path.name,
        progress=progress,
        verbose=verbose,
    )
    records = [
        ChunkRecord.create(
            doc_id=doc_id,
            original_index=next_index + chunk_index,
            chunk_index=chunk_index,
            source=source,
            file_name=path.name,
            file_type=file_type,
            page=chunk.metadata.get("page"),
            extraction=chunk.metadata.get("extraction", "text"),
            heading_path=chunk.metadata.get("heading_path"),
            content=chunk.page_content,
            context=contexts[chunk_index],
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            chunking_strategy=settings.CHUNKING_STRATEGY,
            embedding_model=settings.EMBEDDING_MODEL,
            content_hash=file_hash,
            file_created_at=file_created_at,
            file_modified_at=file_modified_at,
        )
        for chunk_index, chunk in enumerate(chunks)
    ]
    vectors = stages.embed(
        [record.contextualized_content for record in records],
        embeddings=embeddings,
        verbose=verbose,
    )
    # Both stores in one stage boundary (spec §9.6) — ordering rationale in
    # store_chunks.
    stages.store(records, vectors, store=store, bm25=bm25)
    logger.info("%s: ingested %d chunk(s) into both stores", relative, len(records))
    return "ingested", len(records)
