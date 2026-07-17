"""Unit tests for the llama.cpp chat client and clean_response.

The llama.cpp HTTP surface is mocked with respx; request payloads and
tenacity retry behavior are asserted directly, mirroring the embeddings
client's test conventions.
"""

import json
from collections.abc import Iterator
from typing import Any

import httpx
import openai
import pytest
import respx
from openai.types import CompletionUsage
from tenacity import wait_none

from varagity.config import get_settings
from varagity.models.llm import (
    _CTX_HEADROOM_TOKENS,
    GenerationTimings,
    LLMClient,
    clean_response,
)
from varagity.tokens import count_tokens

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


def _stream_chunk(
    *,
    content: str | None = None,
    reasoning_content: str | None = None,
    usage: dict[str, int] | None = None,
    finish: str | None = None,
    timings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    if content is not None:
        delta["content"] = content
    if reasoning_content is not None:
        delta["reasoning_content"] = reasoning_content
    chunk: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test-model.gguf",
        "choices": (
            [] if usage is not None else [{"index": 0, "delta": delta, "finish_reason": finish}]
        ),
    }
    if usage is not None:
        chunk["usage"] = {"total_tokens": sum(usage.values()), **usage}
    if timings is not None:
        # llama.cpp extension: a sibling of `choices`, not nested in `usage`.
        chunk["timings"] = timings
    return chunk


def _stream_response(chunks: list[dict[str, Any]]) -> httpx.Response:
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())


@pytest.fixture
def no_retry_wait() -> Iterator[None]:
    """Zero out the tenacity backoff so retry tests run instantly."""
    retryings = [
        LLMClient._create.retry,  # type: ignore[attr-defined]
        LLMClient._create_stream.retry,  # type: ignore[attr-defined]
    ]
    originals = [retrying.wait for retrying in retryings]
    for retrying in retryings:
        retrying.wait = wait_none()
    yield
    for retrying, original in zip(retryings, originals, strict=True):
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


class TestContextWindowFit:
    """The clamp: prompt + max_tokens must fit LLM_CONTEXT_TOKENS.

    llama.cpp with context shift disabled (its default) hard-fails such
    requests mid-decode instead of stopping at the boundary — the DinoBank
    regression.
    """

    @respx.mock
    def test_oversized_cap_is_clamped_to_fit_the_window(self) -> None:
        route = respx.post(ENDPOINT).mock(return_value=_completion("ok"))
        ctx = get_settings().LLM_CONTEXT_TOKENS
        _client(max_tokens=ctx).generate([{"role": "user", "content": "q"}], verbose=0)
        sent = json.loads(route.calls[0].request.content)
        assert sent["max_tokens"] == ctx - count_tokens("q") - _CTX_HEADROOM_TOKENS

    @respx.mock
    def test_stream_cap_is_clamped_too(self) -> None:
        route = respx.post(ENDPOINT).mock(
            return_value=_stream_response([_stream_chunk(content="x")])
        )
        ctx = get_settings().LLM_CONTEXT_TOKENS
        list(_client(max_tokens=ctx).generate_stream([{"role": "user", "content": "q"}], verbose=0))
        sent = json.loads(route.calls[0].request.content)
        assert sent["max_tokens"] == ctx - count_tokens("q") - _CTX_HEADROOM_TOKENS

    @respx.mock
    def test_within_window_cap_is_untouched(self) -> None:
        route = respx.post(ENDPOINT).mock(return_value=_completion("ok"))
        _client().generate([{"role": "user", "content": "q"}], verbose=0)
        assert json.loads(route.calls[0].request.content)["max_tokens"] == 512

    @respx.mock
    def test_prompt_alone_over_the_window_raises_before_any_request(self) -> None:
        route = respx.post(ENDPOINT).mock(return_value=_completion("ok"))
        huge = "word " * (get_settings().LLM_CONTEXT_TOKENS + 100)
        with pytest.raises(ValueError, match="context window"):
            _client().generate([{"role": "user", "content": huge}], verbose=0)
        assert route.call_count == 0

    @respx.mock
    def test_prompt_counts_every_message(self) -> None:
        """System + user contents both count against the window."""
        route = respx.post(ENDPOINT).mock(return_value=_completion("ok"))
        ctx = get_settings().LLM_CONTEXT_TOKENS
        half = "word " * (ctx // 2)
        with pytest.raises(ValueError, match="context window"):
            _client().generate(
                [
                    {"role": "system", "content": half},
                    {"role": "user", "content": half},
                ],
                verbose=0,
            )
        assert route.call_count == 0


class TestGenerateStream:
    @respx.mock
    def test_yields_deltas_in_order(self) -> None:
        respx.post(ENDPOINT).mock(
            return_value=_stream_response(
                [
                    _stream_chunk(content="The "),
                    _stream_chunk(content="answer."),
                    _stream_chunk(content=None, finish="stop"),
                ]
            )
        )
        deltas = list(_client().generate_stream([{"role": "user", "content": "q"}], verbose=0))
        assert deltas == ["The ", "answer."]

    @respx.mock
    def test_payload_requests_stream_with_usage(self) -> None:
        route = respx.post(ENDPOINT).mock(
            return_value=_stream_response([_stream_chunk(content="x")])
        )
        list(_client().generate_stream([{"role": "user", "content": "q"}], verbose=0))
        sent = json.loads(route.calls[0].request.content)
        assert sent["stream"] is True
        assert sent["stream_options"] == {"include_usage": True}
        assert sent["timings_per_token"] is True
        assert sent["model"] == "test-model.gguf"

    @respx.mock
    def test_think_tags_pass_through_raw(self) -> None:
        respx.post(ENDPOINT).mock(
            return_value=_stream_response(
                [
                    _stream_chunk(content="<think>hmm</think>"),
                    _stream_chunk(content="Answer"),
                ]
            )
        )
        deltas = list(_client().generate_stream([{"role": "user", "content": "q"}], verbose=0))
        assert deltas == ["<think>hmm</think>", "Answer"]

    @respx.mock
    def test_reasoning_content_normalized_to_think_tags(self) -> None:
        """A server extracting reasoning into reasoning_content is re-wrapped."""
        respx.post(ENDPOINT).mock(
            return_value=_stream_response(
                [
                    _stream_chunk(reasoning_content="step one "),
                    _stream_chunk(reasoning_content="step two"),
                    _stream_chunk(content="The answer."),
                ]
            )
        )
        deltas = list(_client().generate_stream([{"role": "user", "content": "q"}], verbose=0))
        assert deltas == ["<think>", "step one ", "step two", "</think>", "The answer."]

    @respx.mock
    def test_reasoning_only_stream_closes_synthesized_tag(self) -> None:
        respx.post(ENDPOINT).mock(
            return_value=_stream_response([_stream_chunk(reasoning_content="thinking…")])
        )
        deltas = list(_client().generate_stream([{"role": "user", "content": "q"}], verbose=0))
        assert deltas == ["<think>", "thinking…", "</think>"]

    @respx.mock
    def test_usage_callback_fires_from_final_chunk(self) -> None:
        respx.post(ENDPOINT).mock(
            return_value=_stream_response(
                [
                    _stream_chunk(content="ok"),
                    _stream_chunk(usage={"prompt_tokens": 11, "completion_tokens": 7}),
                ]
            )
        )
        seen: list[CompletionUsage] = []
        deltas = list(
            _client().generate_stream(
                [{"role": "user", "content": "q"}], verbose=0, on_usage=seen.append
            )
        )
        assert deltas == ["ok"]
        assert len(seen) == 1
        assert (seen[0].prompt_tokens, seen[0].completion_tokens) == (11, 7)

    @respx.mock
    def test_timings_callback_fires_per_carrying_chunk(self) -> None:
        respx.post(ENDPOINT).mock(
            return_value=_stream_response(
                [
                    _stream_chunk(content="a"),  # llama.cpp's first chunk has none
                    _stream_chunk(
                        content="b",
                        timings={"predicted_n": 2, "predicted_ms": 40.0, "cache_n": 81},
                    ),
                    _stream_chunk(
                        content="c",
                        timings={"predicted_n": 3, "predicted_ms": 60.0},
                    ),
                ]
            )
        )
        seen: list[GenerationTimings] = []
        deltas = list(
            _client().generate_stream(
                [{"role": "user", "content": "q"}], verbose=0, on_timings=seen.append
            )
        )
        assert deltas == ["a", "b", "c"]
        assert seen == [
            GenerationTimings(predicted_n=2, predicted_ms=40.0),
            GenerationTimings(predicted_n=3, predicted_ms=60.0),
        ]

    @respx.mock
    def test_timings_never_fires_when_the_server_reports_none(self) -> None:
        # The non-llama.cpp shape: standard chunks, no `timings` anywhere.
        respx.post(ENDPOINT).mock(
            return_value=_stream_response(
                [
                    _stream_chunk(content="ok"),
                    _stream_chunk(usage={"prompt_tokens": 1, "completion_tokens": 1}),
                ]
            )
        )
        seen: list[GenerationTimings] = []
        list(
            _client().generate_stream(
                [{"role": "user", "content": "q"}], verbose=0, on_timings=seen.append
            )
        )
        assert seen == []

    @respx.mock
    def test_malformed_timings_are_skipped_not_fatal(self) -> None:
        respx.post(ENDPOINT).mock(
            return_value=_stream_response(
                [
                    _stream_chunk(content="a", timings={"predicted_n": "not-a-count"}),
                    _stream_chunk(content="b", timings={"predicted_ms": 5.0}),
                ]
            )
        )
        seen: list[GenerationTimings] = []
        deltas = list(
            _client().generate_stream(
                [{"role": "user", "content": "q"}], verbose=0, on_timings=seen.append
            )
        )
        assert deltas == ["a", "b"]
        assert seen == []

    @respx.mock
    def test_establishment_retries_on_5xx_then_streams(self, no_retry_wait: None) -> None:
        route = respx.post(ENDPOINT)
        route.side_effect = [
            httpx.Response(500, json={"error": "boom"}),
            _stream_response([_stream_chunk(content="recovered")]),
        ]
        deltas = list(_client().generate_stream([{"role": "user", "content": "q"}], verbose=0))
        assert deltas == ["recovered"]
        assert route.call_count == 2

    @respx.mock
    def test_establishment_errors_raise_eagerly(self, no_retry_wait: None) -> None:
        """The request is sent (and fails) at call time, not at first next()."""
        route = respx.post(ENDPOINT).mock(return_value=httpx.Response(500, json={"error": "x"}))
        with pytest.raises(openai.InternalServerError):
            _client().generate_stream([{"role": "user", "content": "q"}], verbose=0)
        assert route.call_count == 4  # stop_after_attempt(4)

    @respx.mock
    def test_abandoning_the_iterator_closes_the_stream(self) -> None:
        respx.post(ENDPOINT).mock(
            return_value=_stream_response([_stream_chunk(content="a"), _stream_chunk(content="b")])
        )
        client = _client()
        deltas = client.generate_stream([{"role": "user", "content": "q"}], verbose=0)
        assert next(deltas) == "a"
        deltas.close()  # must not raise; underlying HTTP stream is closed
        assert list(deltas) == []

    @respx.mock
    def test_invalid_verbose_raises_before_any_request(self) -> None:
        route = respx.post(ENDPOINT).mock(
            return_value=_stream_response([_stream_chunk(content="x")])
        )
        with pytest.raises(ValueError, match="verbose"):
            _client().generate_stream([{"role": "user", "content": "q"}], verbose=9)
        assert route.call_count == 0


class TestGenerationTimings:
    def test_rate_is_tokens_over_seconds(self) -> None:
        timings = GenerationTimings(predicted_n=24, predicted_ms=428.028)
        assert timings.tokens_per_second == pytest.approx(56.071, abs=0.001)

    def test_first_chunk_counters_yield_none_not_a_million(self) -> None:
        # llama.cpp's literal first reading: predicted_ms=0.001 computes to
        # 1,000,000 tok/s. predicted_n < 1 is the other degenerate shape.
        assert GenerationTimings(predicted_n=0, predicted_ms=0.0).tokens_per_second is None
        assert GenerationTimings(predicted_n=0, predicted_ms=10.0).tokens_per_second is None
        assert GenerationTimings(predicted_n=5, predicted_ms=0.0).tokens_per_second is None

    def test_single_token_reading_is_still_a_rate(self) -> None:
        # n=1 with real elapsed time is valid arithmetic — the display-side
        # warmup gate, not this property, decides whether to show it.
        assert GenerationTimings(predicted_n=1, predicted_ms=20.0).tokens_per_second == 50.0
