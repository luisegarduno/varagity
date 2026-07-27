"""Offline evaluation harness (spec §16).

Measures retrieval quality — recall@k / pass@k over a hand-authored golden
set — across the four Contextual Retrieval configurations, benchmarks the
pluggable OCR engines, and compares the registered chat engines over
multi-turn conversation fixtures (spec_v3 §4.9). Runs against ephemeral
testcontainers stores (plan decision #4) with the live GPU services for
embeddings/LLM.

The graph half (``eval graph``, spec_graphrag §12) is self-contained by
comparison: the fixture builder writes a synthetic iMessage ``chat.db``
and the graph golden set scores engine *answers* by fact and provenance,
so it needs no stores at all — graph engines self-store in their own
working directories, and the only live services are llama.cpp and
infinity.

Heavy eval-only dependencies (``testcontainers``, ``jiwer`` — the ``eval``
dependency group) are imported at call time, so importing this package
never requires them.
"""

from varagity.eval.datasets import (
    ConversationFixture,
    GoldenEntry,
    GraphGoldenEntry,
    load_conversations,
    load_golden,
    load_graph_golden,
    resolve_golden,
)
from varagity.eval.evaluate import pass_at_k, recall_at_k, run_chat_eval, run_matrix
from varagity.eval.graph_eval import run_graph_eval
from varagity.eval.graph_fixtures import FixtureManifest, build_fixture_chat_db
from varagity.eval.ocr_benchmark import run_ocr_benchmark

__all__ = [
    "ConversationFixture",
    "FixtureManifest",
    "GoldenEntry",
    "GraphGoldenEntry",
    "build_fixture_chat_db",
    "load_conversations",
    "load_golden",
    "load_graph_golden",
    "pass_at_k",
    "recall_at_k",
    "resolve_golden",
    "run_chat_eval",
    "run_graph_eval",
    "run_matrix",
    "run_ocr_benchmark",
]
