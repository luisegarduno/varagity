"""Unit tests for the CLI shell (dispatch, exit codes, the chat loop).

The Prefect flow seams (``ingest_flow``, ``query_flow``) are stubbed at the
CLI module boundary so no Prefect engine (or server) runs here; the flows'
own composition is covered by ``test_pipeline_flows.py`` under the test
harness. The ``query_flow`` stub delegates to the *real*
:func:`~varagity.generation.answer.answer_query`, so the chat tests still
exercise genuine retrieval → grounding → answer wiring.
"""

from collections.abc import Iterator, Sequence

import pytest

from varagity.cli import app as cli_app
from varagity.config import get_settings
from varagity.debug.show import console
from varagity.generation.answer import answer_query
from varagity.ingest.loader import IngestSummary
from varagity.stores.records import RetrievedChunk


@pytest.fixture(autouse=True)
def _fresh_settings() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _chunk(i: int, content: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"docaaa000000000a::{i}",
        doc_id="docaaa000000000a",
        original_index=i,
        content=content,
        context=None,
        metadata={"source": "/docs/corpus/a.md", "file_name": "a.md", "page": None},
        score=0.87 - i / 10,
    )


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int, verbose: int | None = None) -> list[RetrievedChunk]:
        self.calls.append((query, k))
        return self.chunks


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        verbose: int | None = None,
    ) -> str:
        self.prompts.append(messages[0]["content"])
        return self.response


@pytest.fixture
def chat_harness(monkeypatch: pytest.MonkeyPatch, settings_env) -> dict:  # type: ignore[no-untyped-def]
    """Stub every service seam the chat loop touches; script the prompt."""
    settings_env(TOP_K=10, RETRIEVAL_METHOD="semantic")  # hermetic against the machine's .env
    harness: dict = {
        "events": [],
        "retriever": FakeRetriever([_chunk(0, "Lantern produces 4.2 megawatts at peak.")]),
        "llm": FakeLLM("<think>checking…</think>Lantern powers Aurora."),
        "inputs": [":quit"],
        "summary": IngestSummary(discovered=2, ingested=2, chunks=7),
    }

    def fake_ingest(verbose: int) -> IngestSummary:
        harness["events"].append("ingest")
        return harness["summary"]

    def fake_ask(*args: object, **kwargs: object) -> str:
        harness["events"].append("prompt")
        if not harness["inputs"]:
            raise EOFError
        return harness["inputs"].pop(0)

    def fake_query_flow(query: str, **kwargs: object) -> object:
        # The flow's plain twin: same parameters, no Prefect engine.
        return answer_query(query, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(cli_app, "ingest_flow", lambda verbose: fake_ingest(verbose))
    monkeypatch.setattr(cli_app, "query_flow", fake_query_flow)
    monkeypatch.setattr(cli_app, "get_retriever", lambda name: harness["retriever"])
    monkeypatch.setattr(cli_app, "get_model", lambda model_type: harness["llm"])
    monkeypatch.setattr(cli_app.Prompt, "ask", fake_ask)
    return harness


class TestIngestCommand:
    def test_ingest_dispatch_and_exit_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[int] = []

        def fake_ingest(verbose: int, reingest: bool = False) -> IngestSummary:
            calls.append(verbose)
            return IngestSummary(discovered=2, ingested=2, chunks=7)

        monkeypatch.setattr(cli_app, "ingest_flow", fake_ingest)
        with console.capture() as capture:
            exit_code = cli_app.run(["-v", "0", "ingest"])
        assert exit_code == 0
        assert calls == [0]
        out = capture.get()
        assert "Ingest summary" in out
        assert "7" in out

    def test_ingest_exit_one_on_failures(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            cli_app,
            "ingest_flow",
            lambda verbose, reingest=False: IngestSummary(discovered=1, failed=1),
        )
        with console.capture():
            assert cli_app.run(["ingest"]) == 1

    def test_reingest_flag_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`ingest --reingest` reaches the loader; plain `ingest` stays False."""
        seen: list[bool] = []
        monkeypatch.setattr(
            cli_app,
            "ingest_flow",
            lambda verbose, reingest=False: seen.append(reingest) or IngestSummary(),
        )
        with console.capture():
            assert cli_app.run(["ingest"]) == 0
            assert cli_app.run(["ingest", "--reingest"]) == 0
            assert cli_app.run(["ingest", "--reingest", "-v", "0"]) == 0
        assert seen == [False, True, True]

    def test_verbose_flag_overrides_settings_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[int] = []
        monkeypatch.setattr(
            cli_app,
            "ingest_flow",
            lambda verbose, reingest=False: seen.append(verbose) or IngestSummary(),
        )
        with console.capture():
            cli_app.run(["-v", "2", "ingest"])
        assert seen == [2]

    def test_verbose_flag_accepted_after_subcommand(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The plan's canonical invocation is `main.py ingest -v 1`."""
        seen: list[int] = []
        monkeypatch.setattr(
            cli_app,
            "ingest_flow",
            lambda verbose, reingest=False: seen.append(verbose) or IngestSummary(),
        )
        with console.capture():
            cli_app.run(["ingest", "-v", "0"])
            cli_app.run(["-v", "2", "ingest"])  # pre-subcommand value survives
        assert seen == [0, 2]


class TestChatCommand:
    def test_no_subcommand_defaults_to_chat(self, chat_harness: dict) -> None:
        with console.capture() as capture:
            assert cli_app.run(["-v", "0"]) == 0
        out = capture.get()
        assert "Ingest summary" in out  # startup sequence ran
        assert "Ask a question" in out
        assert chat_harness["events"] == ["ingest", "prompt"]

    def test_explicit_chat_subcommand(self, chat_harness: dict) -> None:
        with console.capture():
            assert cli_app.run(["chat", "-v", "0"]) == 0
        assert chat_harness["events"] == ["ingest", "prompt"]

    def test_question_flows_matches_table_then_answer(self, chat_harness: dict) -> None:
        chat_harness["inputs"] = ["What powers Aurora?", ":quit"]
        with console.capture() as capture:
            assert cli_app.run(["-v", "0"]) == 0
        out = capture.get()

        # retrieval happened with the settings TOP_K
        assert chat_harness["retriever"].calls == [("What powers Aurora?", 10)]
        # matches table shows source, score, snippet
        assert "Top 1 matches" in out
        assert "a.md" in out
        assert "0.8700" in out
        assert "Lantern produces 4.2" in out
        # the grounded prompt reached the LLM; the think-stripped answer rendered
        assert "using ONLY the CONTEXT" in chat_harness["llm"].prompts[0]
        assert "Lantern powers Aurora." in out
        assert "<think>" not in out

    def test_quit_exits_cleanly_without_querying(self, chat_harness: dict) -> None:
        chat_harness["inputs"] = [":quit"]
        with console.capture():
            assert cli_app.run(["-v", "0"]) == 0
        assert chat_harness["retriever"].calls == []

    def test_eof_exits_cleanly(self, chat_harness: dict) -> None:
        """Non-interactive stdin (the compose app container) must not crash."""
        chat_harness["inputs"] = []  # first ask raises EOFError
        with console.capture():
            assert cli_app.run(["-v", "0"]) == 0
        assert chat_harness["retriever"].calls == []

    def test_blank_input_is_skipped(self, chat_harness: dict) -> None:
        chat_harness["inputs"] = ["", "   ", ":quit"]
        with console.capture():
            assert cli_app.run(["-v", "0"]) == 0
        assert chat_harness["retriever"].calls == []

    def test_ingest_failures_do_not_block_the_loop(self, chat_harness: dict) -> None:
        chat_harness["summary"] = IngestSummary(discovered=2, ingested=1, failed=1, chunks=3)
        chat_harness["inputs"] = ["still answerable?", ":quit"]
        with console.capture():
            assert cli_app.run(["-v", "0"]) == 0
        assert chat_harness["retriever"].calls == [("still answerable?", 10)]
