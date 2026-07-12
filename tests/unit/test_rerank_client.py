"""Unit tests for the infinity rerank client (spec_v2 §5.4).

The infinity ``/rerank`` HTTP surface is mocked with respx; the Cohere-
protocol request/response schema, the tenacity retry posture, and the
cross-encoder-only rejection are asserted against actual request payloads.
"""

import json
from collections.abc import Iterator

import httpx
import pytest
import respx
from tenacity import wait_none

from varagity.models.rerank import RerankClient, RerankResult

BASE_URL = "http://fake-infinity/v1"
ENDPOINT = f"{BASE_URL}/rerank"

# infinity's actual rejection message for a bi-encoder at /rerank
# (github.com/michaelfeil/infinity discussion #228).
NOT_A_RERANKER = "the loaded model cannot fulfill 'rerank'. options are {'embed'}"


def _client() -> RerankClient:
    return RerankClient(base_url=BASE_URL, api_key="test-key", model="test-reranker")


def _ok_response(request: httpx.Request) -> httpx.Response:
    """Score documents by reverse input position (last doc most relevant)."""
    payload = json.loads(request.content)
    results = [{"index": i, "relevance_score": float(i)} for i in range(len(payload["documents"]))]
    results.sort(key=lambda item: item["relevance_score"], reverse=True)
    if payload.get("top_n") is not None:
        results = results[: payload["top_n"]]
    return httpx.Response(
        200,
        json={
            "object": "rerank",
            "results": results,
            "model": payload["model"],
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        },
    )


@pytest.fixture
def no_retry_wait() -> Iterator[None]:
    """Zero out the tenacity backoff so retry tests run instantly."""
    retrying = RerankClient._post.retry  # type: ignore[attr-defined]
    original = retrying.wait
    retrying.wait = wait_none()
    yield
    retrying.wait = original


class TestRequestSchema:
    @respx.mock
    def test_payload_carries_model_query_documents(self) -> None:
        route = respx.post(ENDPOINT).mock(side_effect=_ok_response)
        _client().rerank("what powers Aurora?", ["doc a", "doc b"], verbose=0)
        sent = json.loads(route.calls[0].request.content)
        assert sent["model"] == "test-reranker"
        assert sent["query"] == "what powers Aurora?"
        assert sent["documents"] == ["doc a", "doc b"]
        assert sent["return_documents"] is False
        assert "top_n" not in sent  # omitted → server scores every document

    @respx.mock
    def test_top_n_passed_through(self) -> None:
        route = respx.post(ENDPOINT).mock(side_effect=_ok_response)
        results = _client().rerank("q", ["a", "b", "c"], top_n=2, verbose=0)
        sent = json.loads(route.calls[0].request.content)
        assert sent["top_n"] == 2
        assert len(results) == 2

    @respx.mock
    def test_auth_header_carries_api_key(self) -> None:
        route = respx.post(ENDPOINT).mock(side_effect=_ok_response)
        _client().rerank("q", ["a"], verbose=0)
        assert route.calls[0].request.headers["authorization"] == "Bearer test-key"

    @respx.mock
    def test_empty_documents_make_no_request(self) -> None:
        route = respx.post(ENDPOINT).mock(side_effect=_ok_response)
        assert _client().rerank("q", [], verbose=0) == []
        assert route.call_count == 0


class TestResponseOrderingMap:
    @respx.mock
    def test_results_map_back_to_input_positions(self) -> None:
        """`index` is the 0-based position in the input list (Cohere protocol)."""
        respx.post(ENDPOINT).mock(side_effect=_ok_response)
        results = _client().rerank("q", ["worst", "middle", "best"], verbose=0)
        assert results == [
            RerankResult(index=2, relevance_score=2.0),
            RerankResult(index=1, relevance_score=1.0),
            RerankResult(index=0, relevance_score=0.0),
        ]

    @respx.mock
    def test_server_order_is_preserved(self) -> None:
        """The client returns the server's ordering verbatim (no re-sort)."""
        respx.post(ENDPOINT).mock(
            return_value=httpx.Response(
                200,
                json={
                    "object": "rerank",
                    "results": [
                        {"index": 1, "relevance_score": 0.9},
                        {"index": 0, "relevance_score": 0.1},
                    ],
                    "model": "m",
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                },
            )
        )
        results = _client().rerank("q", ["a", "b"], verbose=0)
        assert [result.index for result in results] == [1, 0]


class TestRetries:
    @respx.mock
    def test_retries_on_5xx_then_succeeds(self, no_retry_wait: None) -> None:
        route = respx.post(ENDPOINT)
        route.side_effect = [
            httpx.Response(500, json={"error": "boom"}),
            _ok_response,
        ]
        results = _client().rerank("q", ["a"], verbose=0)
        assert len(results) == 1
        assert route.call_count == 2

    @respx.mock
    def test_retry_exhaustion_reraises_http_error(self, no_retry_wait: None) -> None:
        route = respx.post(ENDPOINT).mock(return_value=httpx.Response(500, json={"error": "x"}))
        with pytest.raises(httpx.HTTPStatusError):
            _client().rerank("q", ["a"], verbose=0)
        assert route.call_count == 4  # stop_after_attempt(4)

    @respx.mock
    def test_connect_errors_are_retried(self, no_retry_wait: None) -> None:
        route = respx.post(ENDPOINT)
        route.side_effect = [
            httpx.ConnectError("refused"),
            _ok_response,
        ]
        results = _client().rerank("q", ["a"], verbose=0)
        assert len(results) == 1
        assert route.call_count == 2


class TestCrossEncoderOnly:
    @respx.mock
    def test_non_cross_encoder_rejection_is_not_retried(self, no_retry_wait: None) -> None:
        """A bi-encoder misconfig fails the request once, with a clear message."""
        route = respx.post(ENDPOINT).mock(
            return_value=httpx.Response(400, json={"error": NOT_A_RERANKER})
        )
        with pytest.raises(ValueError, match="cross-encoder"):
            _client().rerank("q", ["a"], verbose=0)
        assert route.call_count == 1

    @respx.mock
    def test_rejection_message_names_model_and_server_error(self) -> None:
        respx.post(ENDPOINT).mock(return_value=httpx.Response(400, json={"error": NOT_A_RERANKER}))
        with pytest.raises(ValueError, match="test-reranker") as exc_info:
            _client().rerank("q", ["a"], verbose=0)
        assert "cannot fulfill" in str(exc_info.value)  # server's reason surfaced


class TestVerbose:
    @respx.mock
    def test_invalid_verbose_raises_before_any_request(self) -> None:
        route = respx.post(ENDPOINT).mock(side_effect=_ok_response)
        with pytest.raises(ValueError, match="verbose"):
            _client().rerank("q", ["a"], verbose=5)
        assert route.call_count == 0


class TestSettingsDefaults:
    def test_defaults_resolve_from_settings(self, settings_env) -> None:  # type: ignore[no-untyped-def]
        settings_env(
            RERANK_MODEL="BAAI/bge-reranker-v2-m3",
            RERANK_API_URL="http://somewhere:8081/v1/",
            RERANK_API_KEY="k",
        )
        client = RerankClient()
        assert client.model == "BAAI/bge-reranker-v2-m3"
        assert client._base_url == "http://somewhere:8081/v1"  # trailing slash stripped
