"""llama.cpp chat client and reasoning-model response cleanup (spec §12).

The llama.cpp server speaks the OpenAI ``/v1`` surface, so the ``openai`` SDK
pointed at ``BASE_MODEL_API_URL`` is the client. Responses from reasoning
models carry ``<think>…</think>`` blocks; callers strip them with
:func:`clean_response` (answers in Phase 4, context blurbs in Phase 5).
"""

import logging
import re
from collections.abc import Sequence

import openai
from openai.types.chat import ChatCompletionMessageParam
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from varagity.config import get_settings
from varagity.debug.show import check_verbose

logger = logging.getLogger(__name__)

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
            ValueError: If ``verbose`` is invalid.
            openai.APIError: If the request still fails after retries.
        """
        check_verbose(get_settings().DEFAULT_VERBOSE if verbose is None else verbose)
        return self._create(
            messages,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            temperature=self.temperature if temperature is None else temperature,
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
