"""Unit tests for varagity.debug.show."""

from pathlib import Path

import pytest
from langchain_core.documents import Document

from varagity.debug.show import (
    VERBOSE_LEVELS,
    check_verbose,
    console,
    trace_badges,
    v_chunk,
    v_discover,
    v_retrieve,
)
from varagity.ingest.discovery import Buckets
from varagity.stores.records import RetrievalTrace, RetrievedChunk


def test_supported_levels_are_0_1_2() -> None:
    assert VERBOSE_LEVELS == (0, 1, 2)


@pytest.mark.parametrize("level", [0, 1, 2])
def test_valid_level_passes_through(level: int) -> None:
    assert check_verbose(level) == level


@pytest.mark.parametrize("level", [-1, 3, 42])
def test_out_of_range_level_raises(level: int) -> None:
    with pytest.raises(ValueError, match="verbose must be one of"):
        check_verbose(level)


def test_non_int_level_raises() -> None:
    with pytest.raises(ValueError, match="verbose must be one of"):
        check_verbose("1")  # type: ignore[arg-type]


class TestVDiscover:
    def _buckets(self) -> Buckets:
        return Buckets(
            text_like=[Path("/docs/a.txt"), Path("/docs/b.md")],
            pdf=[Path("/docs/c.pdf")],
        )

    def test_level_0_renders_nothing(self) -> None:
        with console.capture() as capture:
            v_discover(self._buckets(), verbose=0)
        assert capture.get() == ""

    def test_level_1_shows_counts(self) -> None:
        with console.capture() as capture:
            v_discover(self._buckets(), verbose=1)
        out = capture.get()
        assert "3 document(s)" in out
        assert "2 text-like" in out
        assert "1 pdf" in out
        assert "a.txt" not in out  # file list is level 2

    def test_level_2_lists_files(self) -> None:
        with console.capture() as capture:
            v_discover(self._buckets(), verbose=2)
        out = capture.get()
        assert "a.txt" in out
        assert "c.pdf" in out

    def test_invalid_level_raises(self) -> None:
        with pytest.raises(ValueError, match="verbose"):
            v_discover(self._buckets(), verbose=-2)


class TestVChunk:
    def _chunks(self) -> list[Document]:
        meta = {"file_name": "a.md", "source": "/docs/a.md", "file_type": "md", "page": None}
        return [
            Document(page_content=f"chunk body {i}", metadata={**meta, "chunk_index": i})
            for i in range(2)
        ]

    def test_level_0_renders_nothing(self) -> None:
        with console.capture() as capture:
            v_chunk(self._chunks(), verbose=0)
        assert capture.get() == ""

    def test_level_1_shows_file_and_count(self) -> None:
        with console.capture() as capture:
            v_chunk(self._chunks(), verbose=1)
        out = capture.get()
        assert "a.md" in out
        assert "2 chunk(s)" in out
        assert "chunk body 0" not in out  # panels are level 2

    def test_level_2_renders_chunk_panels(self) -> None:
        with console.capture() as capture:
            v_chunk(self._chunks(), verbose=2)
        out = capture.get()
        assert "chunk body 0" in out
        assert "chunk body 1" in out
        assert "file_type=md" in out

    def test_empty_chunk_list_renders_nothing(self) -> None:
        with console.capture() as capture:
            v_chunk([], verbose=2)
        assert capture.get() == ""


class TestTraceBadges:
    def test_full_trace_renders_all_badges(self) -> None:
        trace = RetrievalTrace(
            semantic_rank=1,
            semantic_score=0.91,
            bm25_rank=3,
            bm25_score=7.5,
            fused_score=0.94,
            fused_rank=2,
            rerank_score=0.88,
            rerank_delta=+2,
            final_rank=1,
        )
        assert trace_badges(trace) == "sem #1 · bm25 #3 · fused 0.94 · rerank +2"

    def test_missing_arm_shows_dash(self) -> None:
        trace = RetrievalTrace(
            bm25_rank=1, bm25_score=7.5, fused_score=7.5, fused_rank=1, final_rank=1
        )
        assert trace_badges(trace) == "sem — · bm25 #1 · fused 7.50"

    def test_negative_delta_keeps_its_sign(self) -> None:
        trace = RetrievalTrace(
            semantic_rank=1,
            semantic_score=0.9,
            fused_score=0.8,
            fused_rank=1,
            rerank_score=0.1,
            rerank_delta=-2,
            final_rank=3,
        )
        assert "rerank -2" in trace_badges(trace)

    def test_no_rerank_stage_omits_the_badge(self) -> None:
        trace = RetrievalTrace(
            semantic_rank=1, semantic_score=0.9, fused_score=0.8, fused_rank=1, final_rank=1
        )
        assert "rerank" not in trace_badges(trace)


class TestVRetrieve:
    def _chunks(
        self, *, context: str | None = None, trace: RetrievalTrace | None = None
    ) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id=f"docaaa000000000a::{i}",
                doc_id="docaaa000000000a",
                original_index=i,
                content=f"retrieved body {i}",
                context=context,
                metadata={"source": "/docs/a.md", "file_name": "a.md", "page": None},
                score=0.91 - i / 10,
                trace=trace,
            )
            for i in range(2)
        ]

    def test_level_0_renders_nothing(self) -> None:
        with console.capture() as capture:
            v_retrieve(self._chunks(), verbose=0)
        assert capture.get() == ""

    def test_level_1_shows_count_only(self) -> None:
        with console.capture() as capture:
            v_retrieve(self._chunks(), verbose=1)
        out = capture.get()
        assert "2 chunk(s)" in out
        assert "retrieved body 0" not in out  # panels are level 2

    def test_level_2_renders_score_source_content_panels(self) -> None:
        with console.capture() as capture:
            v_retrieve(self._chunks(), verbose=2)
        out = capture.get()
        assert "retrieved body 0" in out
        assert "retrieved body 1" in out
        assert "0.9100" in out
        assert "/docs/a.md" in out

    def test_level_2_shows_context_when_present(self) -> None:
        with console.capture() as capture:
            v_retrieve(self._chunks(context="situating blurb"), verbose=2)
        assert "situating blurb" in capture.get()

    def test_level_2_shows_trace_badges_when_present(self) -> None:
        trace = RetrievalTrace(
            semantic_rank=1,
            semantic_score=0.9,
            bm25_rank=3,
            bm25_score=7.5,
            fused_score=0.94,
            fused_rank=1,
            rerank_score=0.88,
            rerank_delta=+2,
            final_rank=1,
        )
        with console.capture() as capture:
            v_retrieve(self._chunks(trace=trace), verbose=2)
        out = capture.get()
        assert "sem #1 · bm25 #3 · fused 0.94 · rerank +2" in out
        assert "retrieved body 0" in out  # content still renders below the badges

    def test_level_2_without_trace_renders_no_badges(self) -> None:
        with console.capture() as capture:
            v_retrieve(self._chunks(), verbose=2)
        assert "sem" not in capture.get()

    def test_invalid_level_raises(self) -> None:
        with pytest.raises(ValueError, match="verbose"):
            v_retrieve(self._chunks(), verbose=3)
