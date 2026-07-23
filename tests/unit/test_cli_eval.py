"""Unit tests for the CLI's eval dispatch and result rendering.

The eval flows themselves need Docker plus the live GPU services, so they
are stubbed at the CLI module boundary (like ``test_cli.py`` does for the
ingest/query flows); these tests cover the ``eval`` subcommand routing and
the table renderers over faithful result documents — the exact shapes
:func:`varagity.eval.evaluate.run_matrix`,
:func:`varagity.eval.evaluate.run_chat_eval`, and
:func:`varagity.eval.ocr_benchmark.run_ocr_benchmark` return.

Documents carry a single k value to keep the tables narrow, and the shared
console is widened for the module so asserted tokens are never wrapped
mid-word by rich's 80-column non-terminal default.
"""

from collections.abc import Iterator
from typing import Any

import pytest

from varagity.cli import app as cli_app
from varagity.config import get_settings
from varagity.debug.show import console


@pytest.fixture(autouse=True)
def _fresh_settings() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _wide_console() -> Iterator[None]:
    saved = console.width
    console.width = 200
    yield
    console.width = saved


def _scores(base: float, k_values: tuple[int, ...]) -> dict[str, Any]:
    return {
        "recall": {str(k): base for k in k_values},
        "pass": {str(k): round(base + 0.005, 3) for k in k_values},
    }


# ── retrieval matrix ─────────────────────────────────────────────────────

_MATRIX_KEYS = (
    "semantic_noncontextual",
    "semantic_contextual",
    "bm25_contextual",
    "hybrid_contextual",
    "hybrid_rerank_contextual",
    "hyde_contextual",
    "hyde_rerank_contextual",
)


def matrix_doc(*, with_sweep: bool = True) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "kind": "retrieval_matrix",
        "n_queries": 11,
        "k_values": [5],
        "chunks_ingested": 42,
        "configs": {key: _scores(0.4 + index / 10, (5,)) for index, key in enumerate(_MATRIX_KEYS)},
        "results_path": "/data/eval/results/20260721-matrix.json",
    }
    if with_sweep:
        doc["chunker_sweep"] = {
            "recursive_character": {
                "chunks": 42,
                "ingest_seconds": 12.34,
                "unresolved_facts": [],
                "configs": {"semantic": _scores(0.61, (5,)), "reranked": _scores(0.71, (5,))},
            },
            "markdown_aware": {
                "chunks": 39,
                "ingest_seconds": 8.9,
                "unresolved_facts": ["stray-fact"],
                "configs": {"semantic": _scores(0.62, (5,)), "reranked": _scores(0.72, (5,))},
            },
        }
    return doc


class TestEvalDispatch:
    def test_eval_runs_matrix_flow_and_renders(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[int] = []
        monkeypatch.setattr(
            cli_app, "eval_flow", lambda verbose: seen.append(verbose) or matrix_doc()
        )
        with console.capture() as capture:
            assert cli_app.run(["-v", "0", "eval"]) == 0
        assert seen == [0]
        out = capture.get()
        assert "Retrieval matrix" in out
        assert "11 golden queries" in out
        assert "42 chunks" in out
        # All seven ladder rungs rendered with their scores.
        assert "0.400" in out and "1.000" in out
        assert "HyDE" in out
        assert "recall@5" in out and "pass@5" in out
        assert "20260721-matrix.json" in out

    def test_eval_renders_result_docs_predating_the_hyde_configs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 5-config document (pre-ADR-016) renders without its missing rows."""
        doc = matrix_doc(with_sweep=False)
        for key in ("hyde_contextual", "hyde_rerank_contextual"):
            del doc["configs"][key]
        monkeypatch.setattr(cli_app, "eval_flow", lambda verbose: doc)
        with console.capture() as capture:
            assert cli_app.run(["-v", "0", "eval"]) == 0
        out = capture.get()
        assert "Retrieval matrix" in out
        assert "HyDE" not in out

    def test_eval_ocr_runs_benchmark_flow(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[int] = []
        monkeypatch.setattr(
            cli_app, "ocr_benchmark_flow", lambda verbose: seen.append(verbose) or ocr_doc()
        )
        with console.capture() as capture:
            assert cli_app.run(["-v", "0", "eval", "ocr"]) == 0
        assert seen == [0]
        assert "OCR intrinsic quality" in capture.get()

    def test_eval_chat_runs_chat_eval_flow(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[int] = []
        monkeypatch.setattr(
            cli_app, "chat_eval_flow", lambda verbose: seen.append(verbose) or chat_doc()
        )
        with console.capture() as capture:
            assert cli_app.run(["-v", "0", "eval", "chat"]) == 0
        assert seen == [0]
        assert "Chat-engine eval" in capture.get()


class TestShowMatrixResults:
    def test_renders_all_five_configs_and_the_sweep(self) -> None:
        with console.capture() as capture:
            cli_app._show_matrix_results(matrix_doc())
        out = capture.get()
        for value in ("0.500", "0.600", "0.700", "0.800", "0.900"):
            assert value in out
        assert "Chunker sweep" in out
        assert "12.3" in out  # recursive_character ingest seconds, .1f
        assert "0.610" in out and "0.720" in out
        # The unresolved-facts warning names the strategy's missing fact.
        assert "stray-fact" in out
        assert "guaranteed misses" in out

    def test_pre_sweep_result_files_render_no_sweep_table(self) -> None:
        """Older matrix documents (no chunker_sweep key) still render."""
        with console.capture() as capture:
            cli_app._show_matrix_results(matrix_doc(with_sweep=False))
        out = capture.get()
        assert "Retrieval matrix" in out
        assert "Chunker sweep" not in out


# ── chat-engine eval ─────────────────────────────────────────────────────

_CHAT_METHODS = ("hybrid", "reranked")


def _engine_summary(base: float) -> dict[str, Any]:
    return {
        method: {
            "all": _scores(base, (1,)),
            "follow_up": _scores(base + 0.05, (1,)),
            # Only the kinds a fixture set actually contains appear; the
            # renderer must skip the absent ones.
            "by_kind": {"pronoun": _scores(base + 0.1, (1,))},
        }
        for method in _CHAT_METHODS
    }


def _turn(
    index: int,
    *,
    kind: str | None,
    query: str,
    search_query: str,
    condensed: bool,
    ranks: dict[str, int | None],
) -> dict[str, Any]:
    return {
        "turn": index,
        "kind": kind,
        "query": query,
        "search_query": search_query,
        "condensed": condensed,
        "methods": {method: {"golden_ranks": dict(ranks)} for method in _CHAT_METHODS},
    }


def _conversations(*, engine_condenses: bool) -> list[dict[str, Any]]:
    follow_up_ranks: dict[str, int | None] = (
        {"depth": 2, "length": None} if engine_condenses else {"depth": None, "length": None}
    )
    return [
        {
            "name": "kelp-corridor",
            "turns": [
                _turn(
                    0,
                    kind=None,
                    query="How long is the kelp corridor?",
                    search_query="How long is the kelp corridor?",
                    condensed=False,
                    ranks={"length": 1},
                ),
                _turn(
                    1,
                    kind="pronoun",
                    query="How deep is it?",
                    search_query=(
                        "How deep is the kelp corridor?" if engine_condenses else "How deep is it?"
                    ),
                    condensed=engine_condenses,
                    ranks=follow_up_ranks,
                ),
            ],
        },
        {
            "name": "reactor",
            "turns": [
                _turn(
                    0,
                    kind=None,
                    query="What powers Aurora?",
                    search_query="What powers Aurora?",
                    condensed=False,
                    ranks={"power": 1},
                ),
                # Neither engine condenses this follow-up: the detail
                # table's "Searched with" column must fall back to a dash.
                _turn(
                    1,
                    kind="topic_shift",
                    query="And the backup system?",
                    search_query="And the backup system?",
                    condensed=False,
                    ranks={"backup": 3},
                ),
            ],
        },
    ]


def chat_doc() -> dict[str, Any]:
    return {
        "kind": "chat_eval",
        "n_conversations": 2,
        "n_turns": 4,
        "n_follow_up_turns": 2,
        "k_values": [1],
        "retrieval_configs": list(_CHAT_METHODS),
        "chunks_ingested": 37,
        "unresolved_facts": ["ghost-fact"],
        "engines": {
            "condense_context": {
                "summary": _engine_summary(0.7),
                "condense": {"calls": 1, "mean_latency_s": 8.612, "max_latency_s": 8.612},
                "conversations": _conversations(engine_condenses=True),
            },
            "simple": {
                "summary": _engine_summary(0.5),
                "condense": {"calls": 0, "mean_latency_s": None, "max_latency_s": None},
                "conversations": _conversations(engine_condenses=False),
            },
        },
        "results_path": "/data/eval/results/20260721-chat.json",
    }


class TestShowChatResults:
    def test_summary_covers_both_methods_and_present_kinds_only(self) -> None:
        with console.capture() as capture:
            cli_app._show_chat_results(chat_doc())
        out = capture.get()
        assert "2 conversations" in out
        assert "4 turns" in out and "2 follow-ups" in out
        assert "all turns" in out and "follow-ups" in out
        assert "pronoun" in out
        # Slices absent from by_kind are skipped, not rendered as blanks.
        assert "elliptical" not in out
        assert "0.700" in out and "0.500" in out  # per-engine "all" recall

    def test_detail_table_ranks_rewrites_and_dashes(self) -> None:
        with console.capture() as capture:
            cli_app._show_chat_results(chat_doc())
        out = capture.get()
        # Follow-ups only, scored under the preferred 'reranked' method.
        assert "reranked" in out
        assert "kelp-corridor" in out and "reactor" in out
        assert "topic_shift" in out
        # condense_context found the golden at rank 2; simple missed (—).
        assert "2" in out and "—" in out
        # The rewrite that drove retrieval is shown.
        assert "How deep is the kelp" in out

    def test_condense_stats_unresolved_facts_and_path(self) -> None:
        with console.capture() as capture:
            cli_app._show_chat_results(chat_doc())
        out = capture.get()
        assert "1 condense call(s)" in out
        assert "8.612" in out
        assert "no condense calls" in out  # the simple engine
        assert "ghost-fact" in out
        assert "20260721-chat.json" in out


# ── OCR benchmark ────────────────────────────────────────────────────────


def ocr_doc() -> dict[str, Any]:
    return {
        "kind": "ocr_benchmark",
        "k_values": [5],
        "fixtures": ["scan-a.pdf", "scan-b.pdf"],
        "n_scanned_queries": 4,
        "engines": {
            "tesseract": {
                "intrinsic": {"overall": {"cer": 0.0421, "wer": 0.1234, "pages_per_sec": 1.52}},
                "retrieval": {
                    "unresolvable_golden_refs": [],
                    "methods": {"semantic": _scores(0.75, (5,)), "hybrid": _scores(0.85, (5,))},
                },
            },
            "easyocr": {
                "intrinsic": {"overall": {"cer": 0.1005, "wer": 0.2311, "pages_per_sec": 0.44}},
                "retrieval": {
                    "unresolvable_golden_refs": ["docbbb0000000000::4"],
                    "methods": {"semantic": _scores(0.5, (5,))},
                },
            },
        },
        "results_path": "/data/eval/results/20260721-ocr.json",
    }


class TestShowOcrResults:
    def test_intrinsic_table_lists_every_engine(self) -> None:
        with console.capture() as capture:
            cli_app._show_ocr_results(ocr_doc())
        out = capture.get()
        assert "2 fixture PDFs" in out
        assert "tesseract" in out and "easyocr" in out
        assert "0.0421" in out and "0.2311" in out
        assert "1.520" in out  # pages/sec, .3f

    def test_retrieval_impact_and_unresolvable_warning(self) -> None:
        with console.capture() as capture:
            cli_app._show_ocr_results(ocr_doc())
        out = capture.get()
        assert "scanned-doc" in out  # the title wraps to the table width
        assert "0.750" in out and "0.850" in out and "0.500" in out
        # easyocr's drifted golden ref is reported as guaranteed misses.
        assert "docbbb0000000000::4" in out
        assert "counted as misses" in out
        assert "20260721-ocr.json" in out
