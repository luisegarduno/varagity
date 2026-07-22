"""Unit tests for the BM25 store's pure logic (real ES ops are integration).

The client-method tests script a fake Elasticsearch client: they verify the
request bodies each method builds (index definition, bulk actions, queries)
and the response parsing, not Elasticsearch behavior itself.
"""

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from elastic_transport import ApiResponseMeta, ConnectionTimeout, HttpHeaders, NodeConfig
from elastic_transport import ConnectionError as ESConnectionError
from elasticsearch import ApiError

from varagity.stores import bm25_store
from varagity.stores.bm25_store import (
    _INDEX_MAPPINGS,
    _INDEX_SETTINGS,
    ElasticsearchBM25,
    _is_transient,
)
from varagity.stores.records import ChunkRecord


def _api_error(status: int) -> ApiError:
    meta = ApiResponseMeta(
        status=status,
        http_version="1.1",
        headers=HttpHeaders({}),
        duration=0.0,
        node=NodeConfig("http", "localhost", 9200),
    )
    return ApiError(message=f"http {status}", meta=meta, body={})


class TestIsTransient:
    def test_connection_trouble_is_transient(self) -> None:
        assert _is_transient(ESConnectionError("refused"))
        assert _is_transient(ConnectionTimeout("timed out"))

    @pytest.mark.parametrize("status", [429, 500, 502, 503])
    def test_throttling_and_5xx_are_transient(self, status: int) -> None:
        assert _is_transient(_api_error(status))

    @pytest.mark.parametrize("status", [400, 401, 404, 409])
    def test_4xx_is_permanent(self, status: int) -> None:
        """Mapping/auth/not-found errors surface immediately, never retried."""
        assert not _is_transient(_api_error(status))

    def test_unrelated_exceptions_are_permanent(self) -> None:
        assert not _is_transient(ValueError("boom"))


class TestIndexDefinition:
    """The spec §8.3 mapping, asserted structurally (live shape: integration)."""

    def test_text_fields_analyzed_with_english(self) -> None:
        properties = _INDEX_MAPPINGS["properties"]
        for field in ("content", "contextualized_content"):
            assert properties[field] == {"type": "text", "analyzer": "english"}

    def test_identity_fields_stored_but_not_indexed(self) -> None:
        properties = _INDEX_MAPPINGS["properties"]
        assert properties["doc_id"] == {"type": "keyword", "index": False}
        assert properties["chunk_id"] == {"type": "keyword", "index": False}
        assert properties["original_index"] == {"type": "integer", "index": False}

    def test_default_analyzer_and_similarity(self) -> None:
        assert _INDEX_SETTINGS["analysis"]["analyzer"]["default"] == {"type": "english"}
        assert _INDEX_SETTINGS["similarity"]["default"] == {"type": "BM25"}


class FakeIndices:
    def __init__(self, *, exists: bool) -> None:
        self._exists = exists
        self.created: list[dict[str, Any]] = []
        self.refreshed: list[str] = []

    def exists(self, index: str) -> bool:
        return self._exists

    def create(self, index: str, settings: Any, mappings: Any) -> None:
        self.created.append({"index": index, "settings": settings, "mappings": mappings})

    def refresh(self, index: str) -> None:
        self.refreshed.append(index)


class FakeESClient:
    def __init__(
        self,
        *,
        exists: bool = False,
        deleted: int = 0,
        search_response: dict[str, Any] | None = None,
    ) -> None:
        self.indices = FakeIndices(exists=exists)
        self._deleted = deleted
        self._search_response = search_response or {"hits": {"hits": []}}
        self.delete_calls: list[dict[str, Any]] = []
        self.search_calls: list[dict[str, Any]] = []
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def delete_by_query(self, *, index: str, query: Any, refresh: bool) -> dict[str, Any]:
        self.delete_calls.append({"index": index, "query": query, "refresh": refresh})
        return {"deleted": self._deleted}

    def search(self, *, index: str, query: Any, size: int) -> dict[str, Any]:
        self.search_calls.append({"index": index, "query": query, "size": size})
        return self._search_response


def store_with(client: FakeESClient, index_name: str = "idx-test") -> ElasticsearchBM25:
    store = ElasticsearchBM25.__new__(ElasticsearchBM25)
    store.index_name = index_name
    store._client = client  # type: ignore[assignment]
    return store


def make_record(index: int) -> ChunkRecord:
    return ChunkRecord(
        doc_id="docaaa000000000a",
        chunk_id=f"docaaa000000000a::{index}",
        original_index=index,
        chunk_index=index,
        source="/abs/corpus/a.md",
        file_name="a.md",
        file_type="md",
        content=f"chunk {index}",
        context="the blurb",
        contextualized_content=f"the blurb\n\nchunk {index}",
        chunk_size=400,
        chunk_overlap=50,
        chunking_strategy="recursive_character",
        embedding_model="intfloat/multilingual-e5-large-instruct",
        n_tokens=4,
        content_hash="deadbeef",
        created_at=datetime(2026, 7, 21, tzinfo=UTC),
    )


class TestLifecycle:
    def test_init_reads_settings_and_disables_sdk_retries(
        self, monkeypatch: pytest.MonkeyPatch, settings_env: Any
    ) -> None:
        settings_env(ELASTICSEARCH_URL="http://es.test:9200", BM25_INDEX_NAME="idx-from-env")
        created: list[dict[str, Any]] = []

        def fake_client(url: str, *, max_retries: int, request_timeout: int) -> FakeESClient:
            created.append(
                {"url": url, "max_retries": max_retries, "request_timeout": request_timeout}
            )
            return FakeESClient()

        monkeypatch.setattr(bm25_store, "Elasticsearch", fake_client)
        store = ElasticsearchBM25()
        assert store.index_name == "idx-from-env"
        assert created == [{"url": "http://es.test:9200", "max_retries": 0, "request_timeout": 30}]
        store.close()
        assert store._client.closed  # type: ignore[attr-defined]

    def test_explicit_arguments_beat_settings(
        self, monkeypatch: pytest.MonkeyPatch, settings_env: Any
    ) -> None:
        settings_env(ELASTICSEARCH_URL="http://es.test:9200", BM25_INDEX_NAME="idx-from-env")
        monkeypatch.setattr(bm25_store, "Elasticsearch", lambda url, **kwargs: FakeESClient())
        store = ElasticsearchBM25(url="http://other:9200", index_name="explicit")
        assert store.index_name == "explicit"


class TestCreateIndex:
    def test_creates_with_the_spec_mapping_when_absent(self) -> None:
        client = FakeESClient(exists=False)
        assert store_with(client).create_index() is True
        (created,) = client.indices.created
        assert created["index"] == "idx-test"
        assert created["settings"] == _INDEX_SETTINGS
        assert created["mappings"] == _INDEX_MAPPINGS

    def test_existing_index_is_left_alone(self) -> None:
        client = FakeESClient(exists=True)
        assert store_with(client).create_index() is False
        assert client.indices.created == []


class TestIndexChunks:
    def test_bulk_actions_address_documents_by_chunk_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bulks: list[list[dict[str, Any]]] = []

        def fake_bulk(client: Any, actions: Any) -> tuple[int, list[Any]]:
            materialized = list(actions)
            bulks.append(materialized)
            return len(materialized), []

        monkeypatch.setattr(bm25_store, "helpers", SimpleNamespace(bulk=fake_bulk))
        client = FakeESClient()
        records = [make_record(0), make_record(1)]
        assert store_with(client).index_chunks(records) == 2
        (actions,) = bulks
        assert [action["_id"] for action in actions] == [
            "docaaa000000000a::0",
            "docaaa000000000a::1",
        ]
        assert actions[0]["_index"] == "idx-test"
        assert actions[0]["_source"]["contextualized_content"] == "the blurb\n\nchunk 0"
        assert actions[0]["_source"]["original_index"] == 0
        # The refresh makes the batch immediately searchable.
        assert client.indices.refreshed == ["idx-test"]

    def test_empty_batch_short_circuits(self) -> None:
        client = FakeESClient()
        assert store_with(client).index_chunks([]) == 0
        assert client.indices.refreshed == []


class TestDeleteDocuments:
    def test_bulk_delete_is_one_terms_query(self) -> None:
        client = FakeESClient(deleted=5)
        assert store_with(client).delete_documents(["d1", "d2"]) == 5
        (call,) = client.delete_calls
        assert call["query"] == {"terms": {"doc_id": ["d1", "d2"]}}
        assert call["refresh"] is True

    def test_single_document_wraps_the_bulk_path(self) -> None:
        client = FakeESClient(deleted=2)
        assert store_with(client).delete_document("d1") == 2
        assert client.delete_calls[0]["query"] == {"terms": {"doc_id": ["d1"]}}

    def test_empty_sequence_skips_the_round_trip(self) -> None:
        client = FakeESClient()
        assert store_with(client).delete_documents([]) == 0
        assert client.delete_calls == []


class TestSearch:
    def test_multi_match_over_both_text_fields(self) -> None:
        response = {
            "hits": {
                "hits": [
                    {
                        "_score": 7.5,
                        "_source": {
                            "doc_id": "docaaa000000000a",
                            "original_index": 3,
                            "content": "chunk text",
                            "contextualized_content": "blurb\n\nchunk text",
                        },
                    }
                ]
            }
        }
        client = FakeESClient(search_response=response)
        (hit,) = store_with(client).search("kelp corridor", k=5, verbose=0)
        assert hit.doc_id == "docaaa000000000a"
        assert hit.original_index == 3
        assert hit.score == 7.5
        (call,) = client.search_calls
        assert call["size"] == 5
        assert call["query"]["multi_match"]["fields"] == ["content", "contextualized_content"]
        assert call["query"]["multi_match"]["query"] == "kelp corridor"

    def test_no_hits_is_an_empty_list(self) -> None:
        assert store_with(FakeESClient()).search("anything", k=5, verbose=0) == []
