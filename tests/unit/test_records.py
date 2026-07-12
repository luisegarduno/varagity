"""Unit tests for varagity.stores.records (id/hash derivation, composition)."""

from datetime import UTC

from varagity.stores.records import (
    ChunkRecord,
    RetrievalTrace,
    RetrievedChunk,
    content_hash,
    derive_doc_id,
)


def _record(**overrides: object) -> ChunkRecord:
    kwargs: dict = {
        "doc_id": "abcd1234abcd1234",
        "original_index": 7,
        "chunk_index": 2,
        "source": "/abs/path/corpus/a.md",
        "file_name": "a.md",
        "file_type": "md",
        "page": None,
        "content": "The reactor produces 4.2 megawatts.",
        "context": None,
        "chunk_size": 400,
        "chunk_overlap": 50,
        "chunking_strategy": "recursive_character",
        "embedding_model": "infloat/multilingual-e5-large-instruct",
        "content_hash": "deadbeef",
    }
    kwargs.update(overrides)
    return ChunkRecord.create(**kwargs)


class TestContentHash:
    def test_is_sha256_hex_of_bytes(self) -> None:
        digest = content_hash(b"hello")
        assert len(digest) == 64
        assert digest == content_hash(b"hello")
        assert digest != content_hash(b"hello!")


class TestDeriveDocId:
    def test_shape_is_16_hex_chars(self) -> None:
        doc_id = derive_doc_id("corpus/a.md", content_hash(b"x"))
        assert len(doc_id) == 16
        assert all(c in "0123456789abcdef" for c in doc_id)

    def test_relative_path_stability(self) -> None:
        """Same relative path + content → same id, regardless of any absolute root.

        This is plan decision #6: absolute paths differ between host and
        container, so they must not participate in the identity.
        """
        file_hash = content_hash(b"same bytes")
        assert derive_doc_id("corpus/a.md", file_hash) == derive_doc_id("corpus/a.md", file_hash)

    def test_different_path_or_content_changes_id(self) -> None:
        file_hash = content_hash(b"same bytes")
        base = derive_doc_id("corpus/a.md", file_hash)
        assert derive_doc_id("corpus/b.md", file_hash) != base
        assert derive_doc_id("corpus/a.md", content_hash(b"other bytes")) != base


class TestChunkRecordCreate:
    def test_chunk_id_shape(self) -> None:
        record = _record()
        assert record.chunk_id == f"{record.doc_id}::{record.chunk_index}"

    def test_identity_composition_without_context(self) -> None:
        """Pre-Phase-5 skeleton invariant (plan decision #1)."""
        record = _record(context=None)
        assert record.context is None
        assert record.contextualized_content == record.content

    def test_composition_with_context(self) -> None:
        """Spec §9.4: contextualized_content = context + blank line + content."""
        record = _record(context="This chunk describes the station's power system.")
        assert record.contextualized_content == (
            "This chunk describes the station's power system.\n\n" + record.content
        )

    def test_n_tokens_counted(self) -> None:
        assert _record().n_tokens > 0
        assert _record(content="a" * 400).n_tokens >= _record(content="a").n_tokens

    def test_created_at_is_utc_aware(self) -> None:
        assert _record().created_at.tzinfo is UTC

    def test_extraction_defaults_to_text(self) -> None:
        assert _record().extraction == "text"
        assert _record(extraction="ocr_fallback").extraction == "ocr_fallback"

    def test_metadata_dump_round_trips_json(self) -> None:
        dumped = _record().model_dump(mode="json")
        assert dumped["page"] is None
        assert isinstance(dumped["created_at"], str)
        assert dumped["extraction"] == "text"


class TestRetrievedChunk:
    def test_carries_score_and_identity(self) -> None:
        retrieved = RetrievedChunk(
            chunk_id="abc::0",
            doc_id="abc",
            original_index=0,
            content="text",
            context=None,
            metadata={"file_name": "a.md"},
            score=0.87,
        )
        assert retrieved.score == 0.87
        assert retrieved.metadata["file_name"] == "a.md"

    def test_trace_defaults_to_none(self) -> None:
        """Backward compatible (spec_v2 §9.2): pre-trace payloads still validate."""
        retrieved = RetrievedChunk(
            chunk_id="abc::0",
            doc_id="abc",
            original_index=0,
            content="text",
            context=None,
            metadata={},
            score=0.5,
        )
        assert retrieved.trace is None


class TestRetrievalTrace:
    def test_minimal_trace_defaults_optional_arms_to_none(self) -> None:
        trace = RetrievalTrace(fused_score=0.8, fused_rank=1, final_rank=1)
        assert trace.semantic_rank is None
        assert trace.bm25_rank is None
        assert trace.rerank_score is None
        assert trace.rerank_delta is None

    def test_full_trace_round_trips(self) -> None:
        """The trace serializes losslessly (the API snapshots it as JSONB later)."""
        trace = RetrievalTrace(
            semantic_rank=1,
            semantic_score=0.91,
            bm25_rank=3,
            bm25_score=7.5,
            fused_score=0.94,
            fused_rank=2,
            rerank_score=0.88,
            rerank_delta=+1,
            final_rank=1,
        )
        assert RetrievalTrace.model_validate(trace.model_dump()) == trace

    def test_attaches_to_retrieved_chunk(self) -> None:
        trace = RetrievalTrace(fused_score=0.8, fused_rank=1, final_rank=1)
        retrieved = RetrievedChunk(
            chunk_id="abc::0",
            doc_id="abc",
            original_index=0,
            content="text",
            context=None,
            metadata={},
            score=0.8,
            trace=trace,
        )
        assert retrieved.trace == trace
