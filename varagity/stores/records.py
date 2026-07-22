"""Chunk-level data model — the canonical metadata record (spec §8.1).

Every chunk carries a complete, validated metadata record. Chunks live in
both stores (pgvector and Elasticsearch) and are joinable by the shared
identity ``(doc_id, original_index)``.

Identity derivation (plan decision #6): ``doc_id`` hashes the path
**relative to** ``DOCS_PATH`` — not the absolute path spec §8.1 sketched —
because absolute paths differ between host and container and across machines,
which would break idempotency and make golden eval sets non-portable. The
absolute path is still recorded in ``source``.
"""

import hashlib
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from varagity.tokens import count_tokens


def content_hash(data: bytes) -> str:
    """Hash raw file bytes for idempotency / dedup (spec §8.2).

    Hashing the bytes (rather than parsed text) lets re-runs skip unchanged
    files *before* paying the parse cost.

    Args:
        data: The file's raw bytes.

    Returns:
        The sha256 hex digest.
    """
    return hashlib.sha256(data).hexdigest()


def derive_doc_id(relative_path: str, file_hash: str) -> str:
    """Derive the stable per-document id (plan decision #6).

    Args:
        relative_path: POSIX-style path of the document **relative to**
            ``DOCS_PATH`` (portable across host/container/machines).
        file_hash: The document's :func:`content_hash`.

    Returns:
        ``sha256(relative_path + ":" + file_hash)`` truncated to 16 hex chars.
    """
    return hashlib.sha256(f"{relative_path}:{file_hash}".encode()).hexdigest()[:16]


class ChunkRecord(BaseModel):
    """One chunk's full metadata record (every spec §8.1 field).

    Attributes:
        doc_id: Stable id per source document (see :func:`derive_doc_id`).
        chunk_id: ``f"{doc_id}::{chunk_index}"`` — the pgvector primary key.
        original_index: Global monotonic chunk index across the corpus (the
            hybrid-fusion identity key, spec §8).
        chunk_index: Chunk position within its document.
        source: Absolute file path (host- or container-local; provenance only).
        file_name: Basename of the source file.
        file_type: ``pdf`` / ``txt`` / ``md``.
        page: Page number (PDF; ``None`` otherwise).
        content: Original chunk text.
        context: LLM-generated situating blurb (``None`` when
            ``CONTEXTUALIZE`` is off — the non-contextual baseline).
        contextualized_content: The text actually embedded and BM25-indexed;
            identical to ``content`` while ``context`` is ``None``.
        chunk_size: Chunk size parameter used, in characters (provenance).
        chunk_overlap: Chunk overlap parameter used, in characters.
        chunking_strategy: Registry name of the chunker used.
        embedding_model: Served embedding model name.
        n_tokens: Approximate token count of ``content`` (plan decision #8).
        content_hash: The parent document's :func:`content_hash`.
        created_at: Ingestion timestamp (UTC) — when the chunk was made,
            not a property of the source file (see ``file_modified_at``).
        file_created_at: Filesystem birth time of the source file (UTC).
            Best-effort: birth time is filesystem/platform-dependent, and a
            copy or download resets it — ``None`` when unavailable.
        file_modified_at: Filesystem mtime of the source file (UTC) — when
            the document's bytes last changed, independent of when they
            were ingested. ``None`` only on rows written before the field
            existed (or if the file vanished mid-ingest).
        extraction: How text was extracted: ``"text"`` (default),
            ``"ocr_fallback"`` (set by the PDF OCR fallback pass), or
            ``"ocr"`` (image parser — OCR is that format's only path) —
            retrieval-quality provenance beyond spec §8.1.
        heading_path: The chunk's markdown heading breadcrumb (e.g.
            ``"Operations > Dredging"``), set by the heading-aware chunkers
            (spec_v2 §7); ``None`` for strategies without structure.
    """

    doc_id: str
    chunk_id: str
    original_index: int
    chunk_index: int
    source: str
    file_name: str
    file_type: str
    page: int | None = None
    content: str
    context: str | None = None
    contextualized_content: str
    chunk_size: int
    chunk_overlap: int
    chunking_strategy: str
    embedding_model: str
    n_tokens: int
    content_hash: str
    created_at: datetime
    file_created_at: datetime | None = None
    file_modified_at: datetime | None = None
    extraction: str = "text"
    heading_path: str | None = None

    @classmethod
    def create(
        cls,
        *,
        doc_id: str,
        original_index: int,
        chunk_index: int,
        source: str,
        file_name: str,
        file_type: str,
        page: int | None,
        content: str,
        context: str | None,
        chunk_size: int,
        chunk_overlap: int,
        chunking_strategy: str,
        embedding_model: str,
        content_hash: str,
        file_created_at: datetime | None = None,
        file_modified_at: datetime | None = None,
        extraction: str = "text",
        heading_path: str | None = None,
    ) -> "ChunkRecord":
        r"""Build a record, deriving the dependent fields.

        Derives ``chunk_id``, ``n_tokens``, ``created_at``, and the
        ``contextualized_content`` composition rule (spec §9.4): with a
        ``context`` blurb it is ``context + "\n\n" + content``; without one
        (``CONTEXTUALIZE`` off, plan decision #2) it is ``content``
        unchanged.

        Args:
            doc_id: Stable id of the parent document.
            original_index: Global monotonic chunk index across the corpus.
            chunk_index: Chunk position within its document.
            source: Absolute file path.
            file_name: Basename of the source file.
            file_type: ``pdf`` / ``txt`` / ``md``.
            page: Page number (PDF; ``None`` otherwise).
            content: Original chunk text.
            context: LLM situating blurb, or ``None`` (``CONTEXTUALIZE`` off).
            chunk_size: Chunk size parameter used (characters).
            chunk_overlap: Chunk overlap parameter used (characters).
            chunking_strategy: Registry name of the chunker used.
            embedding_model: Served embedding model name.
            content_hash: The parent document's content hash.
            file_created_at: Filesystem birth time of the source file, or
                ``None`` when the platform/filesystem doesn't expose one.
            file_modified_at: Filesystem mtime of the source file, or
                ``None`` when it couldn't be read.
            extraction: Extraction provenance (``"text"``, ``"ocr"``, or
                ``"ocr_fallback"``).
            heading_path: Markdown heading breadcrumb, or ``None`` for
                strategies without structure.

        Returns:
            The fully-populated record.
        """
        contextualized = f"{context}\n\n{content}" if context else content
        return cls(
            doc_id=doc_id,
            chunk_id=f"{doc_id}::{chunk_index}",
            original_index=original_index,
            chunk_index=chunk_index,
            source=source,
            file_name=file_name,
            file_type=file_type,
            page=page,
            content=content,
            context=context,
            contextualized_content=contextualized,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunking_strategy=chunking_strategy,
            embedding_model=embedding_model,
            n_tokens=count_tokens(content),
            content_hash=content_hash,
            created_at=datetime.now(UTC),
            file_created_at=file_created_at,
            file_modified_at=file_modified_at,
            extraction=extraction,
            heading_path=heading_path,
        )


class RetrievalTrace(BaseModel):
    """Why one retrieved chunk ranked where it did (spec_v2 §9.2).

    Query-time only — computed per answer by the retrievers, never persisted
    on the immutable chunk rows. Ranks are 1-based (display-ready); an arm's
    rank/score are ``None`` when that arm's ranked list never surfaced the
    chunk. Single-arm retrievers report their arm's score/rank as the fused
    values (there is nothing to fuse).

    Attributes:
        semantic_rank: Rank in the semantic (pgvector) arm.
        semantic_score: Cosine similarity in the semantic arm.
        bm25_rank: Rank in the BM25 (Elasticsearch) arm.
        bm25_score: BM25 relevance in that arm.
        fused_score: Weighted reciprocal-rank fusion score (spec §11.4).
        fused_rank: Rank after fusion.
        rerank_score: Cross-encoder relevance (``None`` when reranking is
            off the path).
        rerank_delta: Positions moved by reranking, ``pre − post`` (+ moved
            up / − moved down; ``None`` when reranking is off the path).
        final_rank: The rank actually returned to the caller.
    """

    semantic_rank: int | None = None
    semantic_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    fused_score: float
    fused_rank: int
    rerank_score: float | None = None
    rerank_delta: int | None = None
    final_rank: int


class RetrievedChunk(BaseModel):
    """A chunk returned by a store search, with its relevance score.

    Field set mirrors the spec §11.2 ``SELECT`` list; ``metadata`` holds the
    full :class:`ChunkRecord` dump persisted at ingest time.

    Attributes:
        chunk_id: The chunk's primary key.
        doc_id: Parent document id.
        original_index: Global chunk index (fusion identity).
        content: Original chunk text.
        context: LLM situating blurb (``None`` when ingested with
            ``CONTEXTUALIZE`` off).
        metadata: Full persisted metadata record.
        score: Similarity score — cosine similarity ``1 - distance`` for the
            vector store (higher is better).
        trace: Rank provenance attached by the retrievers (spec_v2 §9.2);
            ``None`` on raw store results, so pre-trace callers are
            unaffected.
    """

    chunk_id: str
    doc_id: str
    original_index: int
    content: str
    context: str | None
    metadata: dict[str, Any]
    score: float
    trace: RetrievalTrace | None = None


class DocumentInfo(BaseModel):
    """One ingested document, as listed by ``GET /api/documents`` (spec_v2 §4.2).

    A ``documents``-table row joined with its chunks' extraction mix — the
    corpus-management view (file, type, chunk count, ingested-at, how much
    of it came through OCR).

    Attributes:
        doc_id: The document's stable id.
        source: Absolute file path recorded at ingest time.
        file_type: File extension without the dot (``pdf``, ``docx``, …).
        content_hash: sha256 of the source file's bytes at ingest time —
            the preview path re-hashes the on-disk file against it, so an
            edited-but-not-reingested document degrades honestly
            (``file_changed``) instead of previewing the wrong bytes.
        n_chunks: Chunks ingested (``0`` = no extractable text).
        ingested_at: When the document (last) landed in the stores.
        extraction_mix: Chunk count per extraction method (``text`` /
            ``ocr`` / ``ocr_fallback``); empty for a 0-chunk document.
    """

    doc_id: str
    source: str
    file_type: str
    content_hash: str
    n_chunks: int
    ingested_at: datetime
    extraction_mix: dict[str, int]
