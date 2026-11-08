"""Unit tests for the per-corpus write gate (ADR-019)."""

import threading

import pytest

from varagity.api.corpus_gate import CorpusBusy, CorpusGate, get_corpus_gates


class TestCorpusGate:
    def test_a_free_gate_has_no_holder(self) -> None:
        assert CorpusGate("rag").holder is None

    def test_claim_names_the_holder_until_release(self) -> None:
        gate = CorpusGate("rag")
        gate.claim("export")
        assert gate.holder == "export"
        gate.release()
        assert gate.holder is None
        gate.claim("ingest")  # free again: any operation may take it
        assert gate.holder == "ingest"

    def test_a_second_claim_is_refused_without_waiting(self) -> None:
        gate = CorpusGate("rag")
        gate.claim("export")
        with pytest.raises(CorpusBusy) as caught:
            gate.claim("export")  # the same kind is refused too
        assert caught.value.holder == "export"
        assert caught.value.corpus == "rag"
        assert gate.holder == "export"  # the refusal changed nothing

    @pytest.mark.parametrize(
        ("holder", "phrase"),
        [
            ("export", "an export is running"),
            ("ingest", "an ingest is running"),
            ("build", "a build is running"),
            ("restore", "a restore is running"),
        ],
    )
    def test_the_refusal_names_what_to_wait_for(self, holder: str, phrase: str) -> None:
        gate = CorpusGate("graph")
        gate.claim(holder)
        with pytest.raises(CorpusBusy, match=f"the graph corpus is busy — {phrase}"):
            gate.claim("export")

    def test_release_is_idempotent(self) -> None:
        gate = CorpusGate("rag")
        gate.release()
        gate.claim("export")
        gate.release()
        gate.release()
        assert gate.holder is None

    def test_exactly_one_of_many_racing_claims_wins(self) -> None:
        gate = CorpusGate("rag")
        start = threading.Barrier(8)
        winners: list[int] = []
        refused: list[int] = []

        def contend(index: int) -> None:
            start.wait()
            try:
                gate.claim("export")
            except CorpusBusy:
                refused.append(index)
            else:
                winners.append(index)

        threads = [threading.Thread(target=contend, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(winners) == 1
        assert len(refused) == 7


def test_the_process_has_one_gate_per_corpus() -> None:
    gates = get_corpus_gates()
    assert set(gates) == {"rag", "graph"}
    assert gates["rag"].corpus == "rag"
    assert gates["graph"].corpus == "graph"
    assert get_corpus_gates()["rag"] is gates["rag"]  # shared, not re-created
