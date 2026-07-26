"""Per-engine smoke build + query against the live stack (spec_graphrag §8).

Local only, like ``-m integration`` and ``-m e2e``: this layer needs the
``bakeoff`` dependency group installed **and** llama.cpp + infinity running,
so CI never selects it (the default ``addopts`` deselect the marker).

    uv run --group bakeoff pytest -m bakeoff

Each engine builds a graph over the first few dozen scripted fixture
messages and answers one golden question. The assertions are deliberately
about plumbing, not recall — an engine that produces *an* answer with
well-formed evidence has its endpoints, embedding dimensions, storage, and
``<think>`` handling wired correctly, which is what Phase 3's manual gate
checks. Recall is scored by ``eval graph`` in Phase 4, and the numbers land
in ADR-017 in Phase 5.

The wall clock per engine is printed (run with ``-s`` to see it): those
figures are the first real datapoint for scheduling the full-profile runs,
which are expected to take hours per engine at 10⁴ messages on a single slot.
"""

import time
from pathlib import Path

import pytest

from varagity.config import get_settings
from varagity.eval.datasets import GraphGoldenEntry, load_graph_golden
from varagity.eval.graph_fixtures import GRAPH_GOLDEN_PATH, build_fixture_chat_db

# Importing the *package* is what self-registers the adapters (the registry
# idiom) — varagity.graph.base alone would leave the registry empty.
from varagity.graph.engines import GRAPH_ENGINE_REGISTRY, get_graph_engine
from varagity.graph.records import BuildReport, GraphAnswer
from varagity.graph.sources.base import MessageBatch, batch_for_path

pytestmark = pytest.mark.bakeoff

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Enough scripted messages for a real extraction pass, few enough that three
# engines can each index them in a coffee break on a single-slot llama.cpp.
_MESSAGE_LIMIT = 30


@pytest.fixture(scope="module")
def smoke_batch(tmp_path_factory: pytest.TempPathFactory) -> MessageBatch:
    """Build the smoke fixture corpus and parse its first messages back."""
    corpus = tmp_path_factory.mktemp("graph-corpus")
    manifest = build_fixture_chat_db(corpus / "fixture.db", profile="smoke")
    # Sender display names are a settings concern, so a parse without these
    # pins yields raw handles and every golden naming "Bob" would fail.
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("GRAPH_OWNER_ALIASES", manifest.owner_label)
        patch.setenv(
            "GRAPH_HANDLE_NAMES",
            ",".join(f"{handle}={name}" for handle, name in manifest.handle_names.items()),
        )
        get_settings.cache_clear()
        parsed = batch_for_path(corpus / "fixture.db", corpus)
    get_settings.cache_clear()
    return parsed.model_copy(update={"messages": parsed.messages[:_MESSAGE_LIMIT]})


@pytest.fixture(scope="module")
def golden_question(smoke_batch: MessageBatch) -> GraphGoldenEntry:
    """Pick a golden question the smoke subset can actually answer."""
    available = {message.guid for message in smoke_batch.messages}
    entries = load_graph_golden(_REPO_ROOT / GRAPH_GOLDEN_PATH)
    anchored = [entry for entry in entries if set(entry.required_guids) <= available]
    return anchored[0] if anchored else entries[0]


@pytest.mark.parametrize("engine_name", sorted(GRAPH_ENGINE_REGISTRY))
def test_engine_builds_a_graph_and_answers(
    engine_name: str,
    smoke_batch: MessageBatch,
    golden_question: GraphGoldenEntry,
    tmp_path: Path,
) -> None:
    """One engine, end to end: build a tiny graph, then answer one question."""
    workdir = tmp_path / engine_name
    with get_graph_engine(engine_name).session(workdir) as session:
        started = time.perf_counter()
        report = session.build([smoke_batch])
        build_s = time.perf_counter() - started
        stats = session.stats()
        answer = session.query(golden_question.query)

    evidence = answer.evidence
    # The manual gate is a spot-read, so show what came back, not just that it
    # did: the answer text plus an evidence digest, per engine.
    print(
        f"\n[{engine_name}] build {build_s:.1f}s for {report.messages_seen} message(s), "
        f"{len(report.failures)} failure(s); stats={stats.model_dump()}; "
        f"query {answer.latency_s:.1f}s"
    )
    for failure in report.failures:
        print(f"[{engine_name}] build failure: {failure}")
    print(f"[{engine_name}] Q: {golden_question.query}")
    print(f"[{engine_name}] A: {answer.answer}")
    print(
        f"[{engine_name}] evidence: {len(evidence.entities)} entities "
        f"{[entity.name for entity in evidence.entities[:8]]}, "
        f"{len(evidence.relations)} relation(s), {len(evidence.communities)} community(ies), "
        f"{len(evidence.message_guids)} message guid(s)"
    )

    assert isinstance(report, BuildReport)
    assert report.messages_seen == len(smoke_batch.messages)
    assert isinstance(answer, GraphAnswer)
    assert answer.answer.strip(), "the engine produced no answer at all"
    assert "<think>" not in answer.answer, "a reasoning stage leaked into the answer"
    assert answer.mode
    assert all(entity.name for entity in evidence.entities)
    assert all(relation.source or relation.target for relation in evidence.relations)
    assert set(evidence.message_guids) <= {message.guid for message in smoke_batch.messages}
    # Evidence of *something*: an engine that indexed 30 messages and cited no
    # entity, relation, or message either failed extraction or failed
    # retrieval — both are findings worth failing the smoke gate over.
    assert evidence.entities or evidence.relations or evidence.message_guids
