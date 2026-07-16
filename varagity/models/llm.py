"""llama.cpp chat client and reasoning-model response cleanup (spec §12).

The llama.cpp server speaks the OpenAI ``/v1`` surface, so the ``openai`` SDK
pointed at ``BASE_MODEL_API_URL`` is the client. Responses from reasoning
models carry ``<think>…</think>`` blocks; callers strip them with
:func:`clean_response` (answers in Phase 4, context blurbs in Phase 5) or,
on the streaming path (spec_v2 §4.3), classify them delta-by-delta with
:class:`varagity.models.stream.ThinkStreamSplitter`.
"""

import logging
import re
from collections.abc import Callable, Generator, Sequence

import openai
from openai import Stream
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from varagity.config import get_settings
from varagity.debug.show import check_verbose
from varagity.tokens import count_tokens

logger = logging.getLogger(__name__)

# Headroom subtracted when fitting a generation cap into the context window:
# the chat template's scaffolding tokens plus the cl100k approximation's
# drift vs the served model's own tokenizer (~2% measured; plan decision #8).
_CTX_HEADROOM_TOKENS = 512


def _fit_max_tokens(messages: Sequence[ChatCompletionMessageParam], max_tokens: int) -> int:
    """Clamp a generation cap so prompt + generation fits the context window.

    llama.cpp with context shift disabled (its default) fails a request
    mid-decode with a hard 500 ("Context size has been exceeded") once
    prompt + generated tokens reach ``--ctx-size`` — it does not stop
    gracefully at the boundary. Guarantee ``prompt + cap + headroom ≤
    LLM_CONTEXT_TOKENS`` instead: a clamped generation stops with
    ``finish_reason=length``, which callers already handle (an unclosed
    ``<think>`` cleans to an empty response, a cut answer stays an answer).

    Args:
        messages: The chat messages about to be sent (string contents are
            counted; the headroom constant covers template scaffolding).
        max_tokens: The requested generation cap.

    Returns:
        The cap, reduced when needed to fit the window.

    Raises:
        ValueError: If the prompt alone (approximately) overflows the
            window — no generation cap can make the request completable.
    """
    ctx = get_settings().LLM_CONTEXT_TOKENS
    prompt_tokens = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            prompt_tokens += count_tokens(content)
    available = ctx - prompt_tokens - _CTX_HEADROOM_TOKENS
    if available <= 0:
        raise ValueError(
            f"prompt is ~{prompt_tokens} tokens — it cannot fit the model's "
            f"{ctx}-token context window (LLM_CONTEXT_TOKENS) with headroom for "
            "generation"
        )
    if max_tokens > available:
        logger.warning(
            "clamping max_tokens %d → %d: the prompt is ~%d tokens of the %d-token context window",
            max_tokens,
            available,
            prompt_tokens,
            ctx,
        )
        return available
    return max_tokens


# Transient failures worth retrying: connection/timeout trouble, 5xx, and 429.
# 4xx like auth errors are permanent and surface immediately. (Same policy as
# the embeddings client — kept local so each client module stands alone.)
_RETRYABLE_ERRORS = (
    openai.APIConnectionError,  # includes APITimeoutError
    openai.InternalServerError,
    openai.RateLimitError,
)

# A balanced reasoning block anywhere in the response.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# An opener the model never closed (e.g. generation hit MAX_TOKENS mid-think).
_UNCLOSED_THINK_RE = re.compile(r"<think>.*\Z", re.DOTALL)

# Synthesized around `reasoning_content` deltas so the streaming path sees
# one reasoning transport (see LLMClient.generate_stream).
_OPEN_THINK_TAG = "<think>"
_CLOSE_THINK_TAG = "</think>"


def clean_response(text: str) -> str:
    """Strip reasoning-model ``<think>…</think>`` stages from a response.

    Carried from the reference implementation and hardened for the shapes a
    llama.cpp-served reasoning model actually emits:

    * balanced ``<think>…</think>`` blocks (however many) are removed;
    * an orphaned ``</think>`` (some chat templates consume the opening tag)
      drops everything up to and including it;
    * an unclosed ``<think>`` (hit the token cap mid-reasoning) drops
      everything from it onward.

    Args:
        text: The raw LLM response.

    Returns:
        The response without reasoning stages, whitespace-stripped.
    """
    cleaned = _THINK_BLOCK_RE.sub("", text)
    cleaned = _UNCLOSED_THINK_RE.sub("", cleaned)
    # Any closer left now is orphaned: the text before it is reasoning.
    _, closer, after = cleaned.partition("</think>")
    if closer:
        cleaned = after
    return cleaned.strip()


class LLMClient:
    """Chat-completions client for the llama.cpp ``/v1`` endpoint.

    Retries transient failures with exponential backoff (``tenacity`` owns
    retries — the SDK's built-in retry is disabled so behavior is
    single-layered and testable). Generation defaults (``model``,
    ``max_tokens``, ``temperature``) resolve from settings at construction
    and can be overridden per call.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> None:
        """Create the client.

        Args:
            base_url: OpenAI-compatible base URL; defaults to
                ``settings.BASE_MODEL_API_URL``.
            api_key: Bearer token; defaults to ``settings.BASE_MODEL_API_KEY``
                (llama.cpp ignores it, but the SDK requires one).
            model: Model name sent in requests; defaults to
                ``settings.BASE_MODEL`` (llama.cpp serves a single model and
                echoes this field — provenance, not routing).
            max_tokens: Generation cap; defaults to ``settings.MAX_TOKENS``.
            temperature: Sampling temperature; defaults to
                ``settings.LLM_TEMPERATURE``.
        """
        settings = get_settings()
        self.model = model or settings.BASE_MODEL
        self.max_tokens = max_tokens if max_tokens is not None else settings.MAX_TOKENS
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self._client = openai.OpenAI(
            base_url=base_url or settings.BASE_MODEL_API_URL,
            api_key=api_key or settings.BASE_MODEL_API_KEY,
            max_retries=0,  # tenacity owns retries (see class docstring)
        )

    def generate(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        verbose: int | None = None,
    ) -> str:
        """Generate one chat completion.

        The raw response text is returned — reasoning models may include a
        ``<think>…</think>`` stage; callers post-process with
        :func:`clean_response` (spec §10.2, §11.1).

        Args:
            messages: OpenAI-format chat messages.
            max_tokens: Generation cap for this call; defaults to the
                client's ``max_tokens``.
            temperature: Sampling temperature for this call; defaults to the
                client's ``temperature``.
            verbose: Console verbosity (0–2); defaults to
                ``settings.DEFAULT_VERBOSE``.

        Returns:
            The assistant message's text content (empty string if the model
            returned no content).

        Raises:
            ValueError: If ``verbose`` is invalid, or the prompt alone
                overflows ``LLM_CONTEXT_TOKENS``. (The cap is silently
                clamped when *prompt + cap* would overflow — llama.cpp hard-
                fails such requests instead of stopping at the boundary.)
            openai.APIError: If the request still fails after retries.
        """
        check_verbose(get_settings().DEFAULT_VERBOSE if verbose is None else verbose)
        return self._create(
            messages,
            max_tokens=_fit_max_tokens(
                messages, self.max_tokens if max_tokens is None else max_tokens
            ),
            temperature=self.temperature if temperature is None else temperature,
        )

    def generate_stream(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        verbose: int | None = None,
        on_usage: Callable[[CompletionUsage], None] | None = None,
    ) -> Generator[str, None, None]:
        """Generate one chat completion, streamed as raw text deltas.

        The stream's *establishment* (request sent, response headers read) is
        retried like :meth:`generate`; once tokens flow, a mid-stream failure
        surfaces immediately — replaying a half-consumed stream would emit
        duplicate text. Reasoning stages are **not** stripped: ``<think>``
        tags pass through in the deltas for
        :class:`~varagity.models.stream.ThinkStreamSplitter` to classify.
        Servers that extract reasoning into the non-standard
        ``reasoning_content`` delta field instead (llama.cpp under some
        ``--reasoning-format`` settings) are normalized to the same shape:
        those deltas are re-wrapped in synthesized ``<think>…</think>`` tags,
        so downstream sees one contract either way.

        Closing the returned iterator (``.close()``, or abandoning a ``for``
        loop) closes the underlying HTTP stream, which is what aborts a
        llama.cpp generation early — the client-disconnect path (spec_v2
        §4.3 cancellation) relies on it.

        Args:
            messages: OpenAI-format chat messages.
            max_tokens: Generation cap for this call; defaults to the
                client's ``max_tokens``.
            temperature: Sampling temperature for this call; defaults to the
                client's ``temperature``.
            verbose: Console verbosity (0–2); defaults to
                ``settings.DEFAULT_VERBOSE``.
            on_usage: Called with the server-reported token usage when the
                final stream chunk carries it (requested via
                ``stream_options.include_usage``; servers that don't report
                usage simply never trigger it).

        Returns:
            A generator of raw text deltas, in generation order —
            specifically a *generator*: ``close()`` is part of the contract
            (fakes standing in for this method must return one too).

        Raises:
            ValueError: If ``verbose`` is invalid, or the prompt alone
                overflows ``LLM_CONTEXT_TOKENS`` (see :meth:`generate` — the
                same clamp applies here).
            openai.APIError: If establishing the stream still fails after
                retries, or the stream breaks mid-generation.
        """
        check_verbose(get_settings().DEFAULT_VERBOSE if verbose is None else verbose)
        stream = self._create_stream(
            messages,
            max_tokens=_fit_max_tokens(
                messages, self.max_tokens if max_tokens is None else max_tokens
            ),
            temperature=self.temperature if temperature is None else temperature,
        )
        return _iter_stream_deltas(stream, on_usage)

    @retry(
        retry=retry_if_exception_type(_RETRYABLE_ERRORS),
        wait=wait_exponential(multiplier=0.5, max=10),
        stop=stop_after_attempt(4),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _create_stream(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        *,
        max_tokens: int,
        temperature: float,
    ) -> Stream[ChatCompletionChunk]:
        """Open one streaming chat-completions request, retrying establishment.

        Args:
            messages: OpenAI-format chat messages.
            max_tokens: Generation cap for this request.
            temperature: Sampling temperature for this request.

        Returns:
            The established SDK stream (not yet consumed).
        """
        return self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
        )

    @retry(
        retry=retry_if_exception_type(_RETRYABLE_ERRORS),
        wait=wait_exponential(multiplier=0.5, max=10),
        stop=stop_after_attempt(4),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _create(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Send one chat-completions request, retrying transient failures.

        Args:
            messages: OpenAI-format chat messages.
            max_tokens: Generation cap for this request.
            temperature: Sampling temperature for this request.

        Returns:
            The assistant message's text content (``""`` when absent).
        """
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""


def _iter_stream_deltas(
    stream: Stream[ChatCompletionChunk],
    on_usage: Callable[[CompletionUsage], None] | None,
) -> Generator[str, None, None]:
    """Yield text deltas from an established SDK stream, closing it on exit.

    Normalizes the two reasoning transports into one (see
    :meth:`LLMClient.generate_stream`): a non-standard ``reasoning_content``
    delta field is re-wrapped in synthesized ``<think>…</think>`` tags around
    the contiguous reasoning run.

    Args:
        stream: The established streaming response.
        on_usage: Optional callback for the final chunk's token usage.

    Yields:
        Raw text deltas in generation order.
    """
    in_synthesized_think = False
    try:
        for chunk in stream:
            if chunk.usage is not None and on_usage is not None:
                on_usage(chunk.usage)
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            # llama.cpp extension field — absent from the SDK model, so it
            # arrives via pydantic's extra="allow" attribute access.
            reasoning_content = getattr(delta, "reasoning_content", None)
            if reasoning_content:
                if not in_synthesized_think:
                    in_synthesized_think = True
                    yield _OPEN_THINK_TAG
                yield str(reasoning_content)
            if delta.content:
                if in_synthesized_think:
                    in_synthesized_think = False
                    yield _CLOSE_THINK_TAG
                yield delta.content
        if in_synthesized_think:
            # Generation ended while still reasoning: close the synthesized
            # block so the accumulated raw text stays well-formed.
            yield _CLOSE_THINK_TAG
    finally:
        stream.close()
