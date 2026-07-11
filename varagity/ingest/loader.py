r"""Ingestion orchestrator: parse → chunk → contextualize → embed → store (spec §9).

Contextualization (spec §9.4, Phase 5): when ``settings.CONTEXTUALIZE`` is
on, each chunk gets an LLM-generated situating blurb and
``contextualized_content = context + "\n\n" + content`` is what gets
embedded; when off, the identity path (``context = None``,
``contextualized_content = content``) is preserved — the non-contextual eval
baseline (plan decision #2). A document's chunks are contextualized
sequentially, in order, so llama.cpp can reuse its prompt cache across the
shared document preamble.

Dual-write (spec §9.6, Phase 6): every chunk lands in **both** stores —
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
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rich.progress import Progress, TaskID

from varagity.chunking import get_chunker
from varagity.config import Settings, get_settings
from varagity.context import situate_context
from varagity.debug import show
from varagity.debug.show import check_verbose
from varagity.ingest.discovery import discover_documents
from varagity.ingest.parsers import Parser, get_parser
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
_BUCKET_PARSERS: tuple[tuple[str, str], ...] = (("text_like", "text"), ("pdf", "pdf"))


@dataclass
class IngestSummary:
    """Counters for one ingest run, rendered by the CLI summary table.

    Attributes:
        discovered: Files found in the corpus buckets.
        ingested: Files parsed, chunked, embedded, and stored this run.
        skipped: Unchanged files skipped via the idempotency check.
        no_text: Files with no extractable text (recorded as 0-chunk
            documents; includes known-empty files seen again).
        unsupported: Files whose bucket has no registered parser yet
            (PDFs until Phase 7).
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


def ingest_corpus(
    docs_path: str | None = None,
    *,
    store: ContextualVectorDB | None = None,
    bm25: ElasticsearchBM25 | None = None,
    embeddings: EmbeddingsClient | None = None,
    llm: LLMClient | None = None,
    reingest: bool = False,
    verbose: int | None = None,
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

    buckets = discover_documents(str(root), verbose=verbose)
    summary = IngestSummary(discovered=buckets.total)

    # Resolve parsers up front; a bucket without one (PDF until Phase 7) is
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
                "no parser registered for %r — skipping %d file(s); the parser lands "
                "in a later phase",
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
                    )
                except Exception:
                    logger.exception("failed to ingest %s — continuing with the next file", path)
                    summary.failed += 1
                else:
                    if outcome == "ingested":
                        summary.ingested += 1
                    elif outcome == "skipped":
                        summary.skipped += 1
                    else:
                        summary.no_text += 1
                    summary.chunks += n_chunks
                    next_index += n_chunks
                progress.advance(task)
    finally:
        if owns_store:
            active_store.close()
        if owns_bm25:
            active_bm25.close()
    return summary


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

    raw_doc = parser.extract(path, verbose=verbose)
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

    chunker = get_chunker(settings.CHUNKING_STRATEGY)
    chunks = chunker.split(raw_doc.text, source_meta=raw_doc.source_meta, verbose=verbose)
    contexts = _contextualize(
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
            content=chunk.page_content,
            context=contexts[chunk_index],
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            chunking_strategy=settings.CHUNKING_STRATEGY,
            embedding_model=settings.EMBEDDING_MODEL,
            content_hash=file_hash,
        )
        for chunk_index, chunk in enumerate(chunks)
    ]
    vectors = embeddings.embed_passages(
        [record.contextualized_content for record in records], verbose=verbose
    )
    # Both stores in the same boundary (spec §9.6), BM25 first: the pgvector
    # documents row is the idempotency marker and must commit last, so a
    # failure in between leaves the file re-attemptable on the next run
    # (the deterministic chunk_id-addressed BM25 docs then overwrite).
    bm25.index_chunks(records)
    store.store_document(
        doc_id=doc_id,
        source=source,
        file_type=file_type,
        content_hash=file_hash,
        records=records,
        embeddings=vectors,
    )
    logger.info("%s: ingested %d chunk(s) into both stores", relative, len(records))
    return "ingested", len(records)


def _contextualize(
    *,
    document_text: str,
    chunk_texts: list[str],
    llm: LLMClient | None,
    file_name: str,
    progress: Progress,
    verbose: int,
) -> list[str | None]:
    """Generate one situating blurb per chunk (or the identity path).

    Chunks are processed sequentially, in document order, so every call
    shares the same document preamble and llama.cpp reuses its prompt cache
    (spec §9.4). Progress is a per-chunk sub-bar under the run's file bar.

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
