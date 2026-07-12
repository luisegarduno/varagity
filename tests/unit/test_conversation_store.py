"""Unit tests for the conversation store's pure logic.

The SQL round-trips run against real Postgres in the integration suite;
here a scripted fake connection covers the snapshot builder and the
auto-title behavior (LLM cleanup, fallback, only-default-title guard).
"""

from typing import Any

from varagity.stores.conversation_store import (
    DEFAULT_TITLE,
    ConversationStore,
    _generate_title,
    _source_snapshot,
)
from varagity.stores.records import RetrievalTrace, RetrievedChunk


def make_chunk(*, with_trace: bool = True) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="doc::3",
        doc_id="doc",
        original_index=3,
        content="chunk text",
        context="the blurb",
        metadata={
            "source": "/docs/report.pdf",
            "file_name": "report.pdf",
            "file_type": "pdf",
            "page": 4,
            "extraction": "ocr_fallback",
        },
        score=0.87,
        trace=(
            RetrievalTrace(
                semantic_rank=1,
                semantic_score=0.91,
                bm25_rank=3,
                bm25_score=7.2,
                fused_score=0.05,
                fused_rank=1,
                rerank_score=0.98,
                rerank_delta=0,
                final_rank=1,
            )
            if with_trace
            else None
        ),
    )


class TestSourceSnapshot:
    def test_snapshot_carries_everything_the_panel_shows(self) -> None:
        snapshot = _source_snapshot(make_chunk())
        assert snapshot["score"] == 0.87
        assert snapshot["content"] == "chunk text"
        assert snapshot["context"] == "the blurb"
        assert snapshot["source"] == "/docs/report.pdf"
        assert snapshot["file_name"] == "report.pdf"
        assert snapshot["file_type"] == "pdf"
        assert snapshot["page"] == 4
        assert snapshot["extraction"] == "ocr_fallback"
        assert snapshot["trace"]["semantic_rank"] == 1
        assert snapshot["trace"]["rerank_score"] == 0.98

    def test_snapshot_without_trace_is_null(self) -> None:
        assert _source_snapshot(make_chunk(with_trace=False))["trace"] is None


class ScriptedLLM:
    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.calls = 0

    def generate(self, messages: Any, **kwargs: Any) -> str:
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class TestGenerateTitle:
    def test_llm_one_liner_cleaned(self) -> None:
        llm = ScriptedLLM('<think>hmm a title</think>\n"Kelp Corridor Length"\n')
        assert _generate_title("How long is it?", llm=llm) == "Kelp Corridor Length"

    def test_multiline_title_collapsed(self) -> None:
        llm = ScriptedLLM("Kelp\nCorridor   Facts")
        assert _generate_title("q", llm=llm) == "Kelp Corridor Facts"

    def test_llm_failure_falls_back_to_question(self) -> None:
        llm = ScriptedLLM(RuntimeError("llm down"))
        assert _generate_title("  What is the kelp corridor?  ", llm=llm) == (
            "What is the kelp corridor?"
        )

    def test_empty_llm_title_falls_back(self) -> None:
        llm = ScriptedLLM("<think>only reasoning</think>")
        assert _generate_title("The question", llm=llm) == "The question"

    def test_long_results_truncated(self) -> None:
        llm = ScriptedLLM("x" * 500)
        assert len(_generate_title("q", llm=llm)) == 80


class FakeTitleConnection:
    """Scripts the SELECT title row; records UPDATEs."""

    def __init__(self, title: str | None) -> None:
        self.title = title
        self.updates: list[Any] = []
        self.closed = False

    def execute(self, sql: str, params: Any = None) -> Any:
        if sql.startswith("SELECT title"):
            return _Cursor(None if self.title is None else (self.title,))
        if sql.startswith("UPDATE conversations SET title"):
            self.updates.append(params)
            return _Cursor(None)
        raise AssertionError(f"unexpected SQL: {sql}")


class _Cursor:
    def __init__(self, row: Any) -> None:
        self.row = row

    def fetchone(self) -> Any:
        return self.row


def store_with(conn: FakeTitleConnection) -> ConversationStore:
    store = ConversationStore.__new__(ConversationStore)
    store._conn = conn  # type: ignore[assignment]
    return store


class TestAutoTitle:
    def test_titles_a_default_titled_conversation(self) -> None:
        conn = FakeTitleConnection(DEFAULT_TITLE)
        title = store_with(conn).auto_title("c1", "How long?", llm=ScriptedLLM("Kelp Facts"))
        assert title == "Kelp Facts"
        assert conn.updates == [("Kelp Facts", "c1")]

    def test_leaves_an_already_titled_conversation_alone(self) -> None:
        conn = FakeTitleConnection("Custom title")
        llm = ScriptedLLM("New Title")
        assert store_with(conn).auto_title("c1", "q", llm=llm) is None
        assert conn.updates == []
        assert llm.calls == 0

    def test_unknown_conversation_returns_none(self) -> None:
        conn = FakeTitleConnection(None)
        assert store_with(conn).auto_title("nope", "q", llm=ScriptedLLM("T")) is None
        assert conn.updates == []
