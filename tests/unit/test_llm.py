"""Unit tests for the llama.cpp chat client and clean_response.

The llama.cpp HTTP surface is mocked with respx; request payloads and
tenacity retry behavior are asserted directly, mirroring the embeddings
client's test conventions.
"""

import json
from collections.abc import Iterator

import httpx
import openai
import pytest
import respx
from tenacity import wait_none

from varagity.models.llm import LLMClient, clean_response

BASE_URL = "http://fake-llamacpp/v1"
ENDPOINT = f"{BASE_URL}/chat/completions"


def _client(**overrides: object) -> LLMClient:
    defaults: dict = {
        "base_url": BASE_URL,
        "api_key": "none",
        "model": "test-model.gguf",
        "max_tokens": 512,
        "temperature": 0.4,
    }
    defaults.update(overrides)
    return LLMClient(**defaults)


def _completion(content: str | None) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model.gguf",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        },
    )


@pytest.fixture
def no_retry_wait() -> Iterator[None]:
    """Zero out the tenacity backoff so retry tests run instantly."""
    retrying = LLMClient._create.retry  # type: ignore[attr-defined]
    original = retrying.wait
    retrying.wait = wait_none()
    yield
    retrying.wait = original


class TestCleanResponse:
    def test_plain_response_unchanged(self) -> None:
        assert clean_response("The answer is 42.") == "The answer is 42."

    def test_leading_think_block_stripped(self) -> None:
        assert clean_response("<think>hmm, let me see</think>\n\nThe answer.") == "The answer."

    def test_multiline_think_block_stripped(self) -> None:
        raw = "<think>line one\nline two\n\nline three</think>Answer text."
        assert clean_response(raw) == "Answer text."

    def test_multiple_think_blocks_stripped(self) -> None:
        raw = "<think>a</think>First. <think>b</think>Second."
        assert clean_response(raw) == "First. Second."

    def test_unclosed_think_block_dropped_to_end(self) -> None:
        """Generation cut off mid-reasoning (hit MAX_TOKENS) yields no answer."""
        assert clean_response("Partial. <think>still reasoning about") == "Partial."

    def test_orphan_closer_drops_reasoning_before_it(self) -> None:
        """Some chat templates consume the opening tag; only </think> appears."""
        assert clean_response("step 1... step 2...</think>The answer.") == "The answer."

    def test_think_only_response_becomes_empty(self) -> None:
        assert clean_response("<think>nothing but reasoning</think>") == ""

    def test_whitespace_stripped(self) -> None:
        assert clean_response("  spaced out  \n") == "spaced out"


class TestGenerate:
    @respx.mock
    def test_returns_message_content(self) -> None:
        respx.post(ENDPOINT).mock(return_value=_completion("Grounded answer."))
        assert _client().generate([{"role": "user", "content": "q"}], verbose=0) == (
            "Grounded answer."
        )

    @respx.mock
    def test_payload_carries_model_and_defaults(self) -> None:
        route = respx.post(ENDPOINT).mock(return_value=_completion("ok"))
        _client().generate([{"role": "user", "content": "q"}], verbose=0)
        sent = json.loads(route.calls[0].request.content)
        assert sent["model"] == "test-model.gguf"
        assert sent["max_tokens"] == 512
        assert sent["temperature"] == 0.4
        assert sent["messages"] == [{"role": "user", "content": "q"}]

    @respx.mock
    def test_per_call_overrides_win(self) -> None:
        route = respx.post(ENDPOINT).mock(return_value=_completion("ok"))
        _client().generate(
            [{"role": "user", "content": "q"}], max_tokens=64, temperature=0.0, verbose=0
        )
        sent = json.loads(route.calls[0].request.content)
        assert sent["max_tokens"] == 64
        assert sent["temperature"] == 0.0

    @respx.mock
    def test_auth_header_carries_api_key(self) -> None:
        route = respx.post(ENDPOINT).mock(return_value=_completion("ok"))
        _client(api_key="secret-token").generate([{"role": "user", "content": "q"}], verbose=0)
        assert route.calls[0].request.headers["authorization"] == "Bearer secret-token"

    @respx.mock
    def test_null_content_returns_empty_string(self) -> None:
        respx.post(ENDPOINT).mock(return_value=_completion(None))
        assert _client().generate([{"role": "user", "content": "q"}], verbose=0) == ""

    @respx.mock
    def test_raw_think_content_passed_through(self) -> None:
        """generate() returns the raw response; callers own clean_response."""
        respx.post(ENDPOINT).mock(return_value=_completion("<think>x</think>y"))
        assert _client().generate([{"role": "user", "content": "q"}], verbose=0) == (
            "<think>x</think>y"
        )


class TestRetries:
    @respx.mock
    def test_retries_on_5xx_then_succeeds(self, no_retry_wait: None) -> None:
        route = respx.post(ENDPOINT)
        route.side_effect = [
            httpx.Response(500, json={"error": "boom"}),
            _completion("recovered"),
        ]
        assert _client().generate([{"role": "user", "content": "q"}], verbose=0) == "recovered"
        assert route.call_count == 2

    @respx.mock
    def test_retry_exhaustion_reraises(self, no_retry_wait: None) -> None:
        route = respx.post(ENDPOINT).mock(return_value=httpx.Response(500, json={"error": "x"}))
        with pytest.raises(openai.InternalServerError):
            _client().generate([{"role": "user", "content": "q"}], verbose=0)
        assert route.call_count == 4  # stop_after_attempt(4)

    @respx.mock
    def test_auth_errors_are_not_retried(self, no_retry_wait: None) -> None:
        route = respx.post(ENDPOINT).mock(
            return_value=httpx.Response(401, json={"error": "bad key"})
        )
        with pytest.raises(openai.AuthenticationError):
            _client().generate([{"role": "user", "content": "q"}], verbose=0)
        assert route.call_count == 1


class TestVerbose:
    @respx.mock
    def test_invalid_verbose_raises_before_any_request(self) -> None:
        route = respx.post(ENDPOINT).mock(return_value=_completion("ok"))
        with pytest.raises(ValueError, match="verbose"):
            _client().generate([{"role": "user", "content": "q"}], verbose=7)
        assert route.call_count == 0
