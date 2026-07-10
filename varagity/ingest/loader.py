"""Ingestion orchestrator: parse → chunk → embed → store, per file (spec §9).

Phase-3 skeleton semantics (plan decision #1): contextualization does not
exist yet, so every record is built with ``context = None`` and
``contextualized_content = content``. Phase 5 changes only the contextualize
step — no schema or store changes.

Idempotency: a file whose ``(doc_id, content_hash)`` is already recorded is
skipped *before* parsing. A file with no extractable text is never silently
dropped: it gets a ``documents`` row with ``n_chunks = 0``, a warning, and a
dedicated summary count.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rich.progress import Progress

from varagity.chunking import get_chunker
from varagity.config import Settings, get_settings
from varagity.debug import show
from varagity.debug.show import check_verbose
from varagity.ingest.discovery import discover_documents
from varagity.ingest.parsers import Parser, get_parser
from varagity.models import EmbeddingsClient, get_model
from varagity.stores import ChunkRecord, ContextualVectorDB, content_hash, derive_doc_id

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
    embeddings: EmbeddingsClient | None = None,
    verbose: int | None = None,
) -> IngestSummary:
    """Ingest every supported document under ``docs_path``.

    Args:
        docs_path: Corpus directory; defaults to ``settings.DOCS_PATH``.
        store: Vector store to write to; constructed from settings (and
            closed on return) when omitted.
        embeddings: Embeddings client; resolved via the model registry when
            omitted.
        verbose: Console verbosity (0–2); defaults to
            ``settings.DEFAULT_VERBOSE``.

    Returns:
        The run's counters (one file failing is counted and logged, not
        raised — a bad file must not abort a corpus run).

    Raises:
        ValueError: If ``verbose`` is invalid.
        psycopg.OperationalError: If the vector store is unreachable.
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
    try:
        client = embeddings if embeddings is not None else get_model("embedding")
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
                        embeddings=client,
                        settings=settings,
                        next_index=next_index,
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
    return summary


def _ingest_file(
    *,
    path: Path,
    root: Path,
    parser: Parser,
    store: ContextualVectorDB,
    embeddings: EmbeddingsClient,
    settings: Settings,
    next_index: int,
    verbose: int,
) -> tuple[Literal["ingested", "skipped", "no_text"], int]:
    """Ingest a single file.

    Args:
        path: The file to ingest.
        root: The corpus root (``doc_id`` hashes the path relative to it,
            plan decision #6).
        parser: Parser for the file's bucket.
        store: The vector store.
        embeddings: The embeddings client.
        settings: Loaded application settings.
        next_index: First free global ``original_index`` for this file.
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

    if store.document_exists(doc_id, file_hash):
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
            context=None,  # identity until Phase 5 (plan decision #1)
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
    store.store_document(
        doc_id=doc_id,
        source=source,
        file_type=file_type,
        content_hash=file_hash,
        records=records,
        embeddings=vectors,
    )
    logger.info("%s: ingested %d chunk(s)", relative, len(records))
    return "ingested", len(records)
