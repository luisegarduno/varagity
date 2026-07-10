"""Unit tests for the embeddings client (spec §15.2 "models/embeddings" row).

The infinity HTTP surface is mocked with respx; e5 formatting, batching, and
tenacity retry behavior are asserted against the actual request payloads.
"""

import json
import logging
from collections.abc import Iterator

import httpx
import openai
import pytest
import respx
from tenacity import wait_none

from varagity.models.embeddings import (
    E5_QUERY_TASK,
    TOKEN_WARN_THRESHOLD,
    EmbeddingsClient,
    format_query,
)

BASE_URL = "http://fake-infinity/v1"
ENDPOINT = f"{BASE_URL}/embeddings"


def _client(batch_size: int = 2) -> EmbeddingsClient:
    return EmbeddingsClient(
        base_url=BASE_URL, api_key="test-key", model="test-model", batch_size=batch_size
    )


def _ok_response(request: httpx.Request) -> httpx.Response:
    payload = json.loads(request.content)
    data = [
        {"object": "embedding", "index": i, "embedding": [float(i), 0.5, -0.5]}
        for i in range(len(payload["input"]))
    ]
    return httpx.Response(
        200,
        json={
            "object": "list",
            "data": data,
            "model": payload["model"],
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        },
    )


@pytest.fixture
def no_retry_wait() -> Iterator[None]:
    """Zero out the tenacity backoff so retry tests run instantly."""
    retrying = EmbeddingsClient._embed_batch.retry  # type: ignore[attr-defined]
    original = retrying.wait
    retrying.wait = wait_none()
    yield
    retrying.wait = original


class TestE5Formatting:
    def test_format_query_wraps_with_instruction(self) -> None:
        assert format_query("what powers Aurora?") == (
            f"Instruct: {E5_QUERY_TASK}\nQuery: what powers Aurora?"
        )

    @respx.mock
    def test_passages_sent_raw_without_prefix(self) -> None:
        route = respx.post(ENDPOINT).mock(side_effect=_ok_response)
        _client().embed_passages(["passage one", "passage two"], verbose=0)
        sent = json.loads(route.calls[0].request.content)
        assert sent["input"] == ["passage one", "passage two"]

    @respx.mock
    def test_query_sent_instruction_wrapped(self) -> None:
        route = respx.post(ENDPOINT).mock(side_effect=_ok_response)
        _client().embed_query("what powers Aurora?", verbose=0)
        sent = json.loads(route.calls[0].request.content)
        assert sent["input"] == [f"Instruct: {E5_QUERY_TASK}\nQuery: what powers Aurora?"]

    @respx.mock
    def test_auth_header_carries_api_key(self) -> None:
        route = respx.post(ENDPOINT).mock(side_effect=_ok_response)
        _client().embed_query("q", verbose=0)
        assert route.calls[0].request.headers["authorization"] == "Bearer test-key"


class TestBatching:
    @respx.mock
    def test_batches_by_batch_size(self) -> None:
        route = respx.post(ENDPOINT).mock(side_effect=_ok_response)
        vectors = _client(batch_size=2).embed_passages(["a", "b", "c", "d", "e"], verbose=0)
        assert len(vectors) == 5
        assert route.call_count == 3  # 2 + 2 + 1
        sizes = [len(json.loads(call.request.content)["input"]) for call in route.calls]
        assert sizes == [2, 2, 1]

    @respx.mock
    def test_empty_input_makes_no_requests(self) -> None:
        route = respx.post(ENDPOINT).mock(side_effect=_ok_response)
        assert _client().embed_passages([], verbose=0) == []
        assert route.call_count == 0

    @respx.mock
    def test_results_ordered_by_response_index(self) -> None:
        def shuffled(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            data = [
                {"object": "embedding", "index": i, "embedding": [float(i)]}
                for i in range(len(payload["input"]))
            ]
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": list(reversed(data)),  # out of order on purpose
                    "model": "m",
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                },
            )

        respx.post(ENDPOINT).mock(side_effect=shuffled)
        vectors = _client(batch_size=3).embed_passages(["a", "b", "c"], verbose=0)
        assert vectors == [[0.0], [1.0], [2.0]]


class TestRetries:
    @respx.mock
    def test_retries_on_5xx_then_succeeds(self, no_retry_wait: None) -> None:
        route = respx.post(ENDPOINT)
        route.side_effect = [
            httpx.Response(500, json={"error": "boom"}),
            _ok_response,
        ]
        vectors = _client().embed_passages(["a"], verbose=0)
        assert len(vectors) == 1
        assert route.call_count == 2

    @respx.mock
    def test_retry_exhaustion_reraises(self, no_retry_wait: None) -> None:
        route = respx.post(ENDPOINT).mock(return_value=httpx.Response(500, json={"error": "x"}))
        with pytest.raises(openai.InternalServerError):
            _client().embed_passages(["a"], verbose=0)
        assert route.call_count == 4  # stop_after_attempt(4)

    @respx.mock
    def test_auth_errors_are_not_retried(self, no_retry_wait: None) -> None:
        route = respx.post(ENDPOINT).mock(
            return_value=httpx.Response(401, json={"error": "bad key"})
        )
        with pytest.raises(openai.AuthenticationError):
            _client().embed_query("q", verbose=0)
        assert route.call_count == 1


class TestTokenGuard:
    @respx.mock
    def test_warns_when_passage_nears_e5_limit(self, caplog: pytest.LogCaptureFixture) -> None:
        respx.post(ENDPOINT).mock(side_effect=_ok_response)
        long_passage = "station " * TOKEN_WARN_THRESHOLD  # ≥1 token per repetition
        with caplog.at_level(logging.WARNING):
            _client(batch_size=600).embed_passages([long_passage], verbose=0)
        assert any("truncates at 512" in r.message for r in caplog.records)

    @respx.mock
    def test_no_warning_for_short_passages(self, caplog: pytest.LogCaptureFixture) -> None:
        respx.post(ENDPOINT).mock(side_effect=_ok_response)
        with caplog.at_level(logging.WARNING):
            _client().embed_passages(["short and sweet"], verbose=0)
        assert not [r for r in caplog.records if "truncates" in r.message]


class TestVerbose:
    @respx.mock
    def test_invalid_verbose_raises_before_any_request(self) -> None:
        route = respx.post(ENDPOINT).mock(side_effect=_ok_response)
        with pytest.raises(ValueError, match="verbose"):
            _client().embed_passages(["a"], verbose=5)
        assert route.call_count == 0
