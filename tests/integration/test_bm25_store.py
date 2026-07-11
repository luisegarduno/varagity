"""Integration tests for ElasticsearchBM25 against a real Elasticsearch.

Runs a throwaway single-node Elasticsearch (same image as compose, via the
shared :mod:`varagity.eval.containers` helpers) and exercises the spec §8.3
mapping shape, contextual ``multi_match`` ranking, idempotent indexing, and
``delete_by_query`` — including that term queries work against the
``index: false`` identity fields (doc-values scan).

Select with ``pytest -m integration`` (needs Docker).
"""

from collections.abc import Iterator
from typing import Any

import pytest
from elasticsearch import Elasticsearch

from varagity.eval.containers import ephemeral_elasticsearch
from varagity.stores.bm25_store import ElasticsearchBM25
from varagity.stores.records import ChunkRecord

pytestmark = pytest.mark.integration

INDEX = "varagity_test_bm25"


@pytest.fixture(scope="session")
def es_url() -> Iterator[str]:
    """A single-node Elasticsearch for the whole session."""
    with ephemeral_elasticsearch() as url:
        yield url


@pytest.fixture
def raw_client(es_url: str) -> Iterator[Elasticsearch]:
    """A raw client for assertions the store's API deliberately doesn't offer."""
    client = Elasticsearch(es_url)
    yield client
    client.close()


@pytest.fixture
def store(es_url: str, raw_client: Elasticsearch) -> Iterator[ElasticsearchBM25]:
    """A store on a fresh index (deleted and recreated per test).

    The create's return value is deliberately not asserted here: on a cold
    node the create request can time out client-side after the server has
    already created the index, making the retried call report "existed".
    What the fixture (and the loader) rely on is the postcondition — the
    index exists — which the boolean-focused idempotency test checks on a
    warm node.
    """
    raw_client.indices.delete(index=INDEX, ignore_unavailable=True)
    with ElasticsearchBM25(url=es_url, index_name=INDEX) as bm25:
        bm25.create_index()
        assert raw_client.indices.exists(index=INDEX)
        yield bm25


def _record(
    doc_id: str, chunk_index: int, original_index: int, content: str, context: str | None = None
) -> ChunkRecord:
    return ChunkRecord.create(
        doc_id=doc_id,
        original_index=original_index,
        chunk_index=chunk_index,
        source=f"/abs/corpus/{doc_id}.md",
        file_name=f"{doc_id}.md",
        file_type="md",
        page=None,
        content=content,
        context=context,
        chunk_size=400,
        chunk_overlap=50,
        chunking_strategy="recursive_character",
        embedding_model="test-model",
        content_hash="hash-" + doc_id,
    )


def _count(raw_client: Elasticsearch) -> int:
    return int(raw_client.count(index=INDEX)["count"])


class TestIndexShape:
    def test_mapping_matches_spec(
        self, store: ElasticsearchBM25, raw_client: Elasticsearch
    ) -> None:
        """The live index carries the spec §8.3 mapping and settings."""
        mapping: dict[str, Any] = raw_client.indices.get_mapping(index=INDEX)[INDEX]
        properties = mapping["mappings"]["properties"]
        for field in ("content", "contextualized_content"):
            assert properties[field]["type"] == "text"
            assert properties[field]["analyzer"] == "english"
        for field, es_type in (
            ("doc_id", "keyword"),
            ("chunk_id", "keyword"),
            ("original_index", "integer"),
        ):
            assert properties[field]["type"] == es_type
            assert properties[field]["index"] is False

        settings: dict[str, Any] = raw_client.indices.get_settings(index=INDEX)[INDEX]
        index_settings = settings["settings"]["index"]
        assert index_settings["analysis"]["analyzer"]["default"]["type"] == "english"
        assert index_settings["similarity"]["default"]["type"] == "BM25"

    def test_create_index_is_idempotent(
        self, store: ElasticsearchBM25, raw_client: Elasticsearch
    ) -> None:
        assert store.create_index() is False  # already exists: no error, not re-created
        raw_client.indices.delete(index=INDEX)
        assert store.create_index() is True  # warm node: fresh create reports created
        assert store.create_index() is False


class TestSearch:
    def test_planted_document_ranks_first(self, store: ElasticsearchBM25) -> None:
        store.index_chunks(
            [
                _record("docaaa000000000a", 0, 0, "The reactor is refueled every eleven years."),
                _record(
                    "docaaa000000000a",
                    1,
                    1,
                    "A crawler robot named Zephyrion inspects the turbines.",
                ),
                _record("docbbb000000000b", 0, 2, "Kelp dampens turbulence between the arrays."),
            ]
        )

        results = store.search("Zephyrion robot", k=3, verbose=0)

        assert results, "expected at least the planted chunk"
        top = results[0]
        assert top.doc_id == "docaaa000000000a"
        assert top.original_index == 1
        assert "Zephyrion" in top.content
        assert top.score > 0
        # every hit carries the spec §11.3 fields
        for hit in results:
            assert hit.contextualized_content
            assert hit.score > 0

    def test_context_blurb_is_searchable(self, store: ElasticsearchBM25) -> None:
        """Contextual BM25: a term only in the situating blurb still matches."""
        store.index_chunks(
            [
                _record(
                    "docccc000000000c",
                    0,
                    10,
                    "It weighs 19 kilograms.",  # ambiguous chunk on its own
                    context="From the Brinewing maintenance drone datasheet.",
                ),
                _record("docddd000000000d", 0, 11, "The corridor is 1.8 kilometers long."),
            ]
        )

        results = store.search("Brinewing drone", k=2, verbose=0)

        assert results
        assert results[0].doc_id == "docccc000000000c"
        assert "Brinewing" not in results[0].content  # matched via the blurb…
        assert "Brinewing" in results[0].contextualized_content  # …which is indexed

    def test_english_analyzer_stems(self, store: ElasticsearchBM25) -> None:
        store.index_chunks(
            [_record("doceee000000000e", 0, 20, "Eighty-seven underwater turbines spin daily.")]
        )
        results = store.search("turbine", k=1, verbose=0)  # singular query, plural doc
        assert results and results[0].doc_id == "doceee000000000e"

    def test_k_limits_results(self, store: ElasticsearchBM25) -> None:
        store.index_chunks(
            [
                _record("docfff000000000f", i, 30 + i, f"tidal energy fact number {i}")
                for i in range(5)
            ]
        )
        assert len(store.search("tidal energy", k=2, verbose=0)) == 2

    def test_invalid_verbose_raises(self, store: ElasticsearchBM25) -> None:
        with pytest.raises(ValueError, match="verbose"):
            store.search("q", k=1, verbose=9)


class TestIndexing:
    def test_index_chunks_is_idempotent_by_chunk_id(
        self, store: ElasticsearchBM25, raw_client: Elasticsearch
    ) -> None:
        """Re-indexing the same chunks overwrites instead of duplicating."""
        records = [_record("docggg000000000g", i, 40 + i, f"content {i}") for i in range(3)]
        assert store.index_chunks(records) == 3
        assert store.index_chunks(records) == 3  # same _ids → overwrite
        assert _count(raw_client) == 3

    def test_empty_batch_is_a_noop(self, store: ElasticsearchBM25) -> None:
        assert store.index_chunks([]) == 0


class TestDeletion:
    def test_delete_document_removes_only_that_doc(
        self, store: ElasticsearchBM25, raw_client: Elasticsearch
    ) -> None:
        """`--reingest` backing: delete_by_query works on the index:false doc_id."""
        store.index_chunks(
            [
                _record("docdel000000000d", 0, 50, "delete me"),
                _record("docdel000000000d", 1, 51, "delete me too"),
                _record("dockeep00000000k", 0, 52, "keep me"),
            ]
        )
        assert store.delete_document("docdel000000000d") == 2
        assert _count(raw_client) == 1
        remaining = store.search("keep", k=5, verbose=0)
        assert [hit.doc_id for hit in remaining] == ["dockeep00000000k"]

    def test_delete_unknown_document_is_a_noop(self, store: ElasticsearchBM25) -> None:
        assert store.delete_document("doc0000000000nil") == 0
