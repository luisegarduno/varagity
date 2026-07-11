"""Unit tests for the BM25 store's pure logic (real ES ops are integration)."""

import pytest
from elastic_transport import ApiResponseMeta, ConnectionTimeout, HttpHeaders, NodeConfig
from elastic_transport import ConnectionError as ESConnectionError
from elasticsearch import ApiError

from varagity.stores.bm25_store import _INDEX_MAPPINGS, _INDEX_SETTINGS, _is_transient


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
