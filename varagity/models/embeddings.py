r"""Infinity embeddings client with the two e5-instruct modes.

``multilingual-e5-large-instruct`` formats **asymmetrically** (spec §9.5,
research §2.3), and getting it wrong degrades recall *silently*:

* **Passages** (ingestion) are embedded as raw text — no prefix. The model
  card is explicit: "No need to add instruction for retrieval documents."
* **Queries** are wrapped as ``Instruct: {task}\nQuery: {query}``.

Both modes live here so no caller can format inconsistently.
"""

import logging

import openai
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

# The e5-large-instruct model card's default retrieval task, used verbatim.
E5_QUERY_TASK = "Given a web search query, retrieve relevant passages that answer the query"

# e5 truncates at 512 tokens; warn near the limit so config drift (bigger
# chunks, long context blurbs) is visible at ingest time.
TOKEN_WARN_THRESHOLD = 480

# Transient failures worth retrying: connection/timeout trouble, 5xx, and 429.
# 4xx like auth errors are permanent and surface immediately.
_RETRYABLE_ERRORS = (
    openai.APIConnectionError,  # includes APITimeoutError
    openai.InternalServerError,
    openai.RateLimitError,
)


def format_query(query: str) -> str:
    r"""Wrap a query in the e5-instruct format (query mode).

    Args:
        query: The user's search query, unformatted.

    Returns:
        ``Instruct: {task}\nQuery: {query}`` with the model card's default
        retrieval task.
    """
    return f"Instruct: {E5_QUERY_TASK}\nQuery: {query}"


class EmbeddingsClient:
    """Client for the infinity ``/v1/embeddings`` endpoint.

    Batches requests by ``EMBEDDING_BATCH_SIZE`` and retries transient
    failures with exponential backoff (``tenacity`` owns retries — the SDK's
    built-in retry is disabled so behavior is single-layered and testable).
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        """Create the client.

        Args:
            base_url: OpenAI-compatible base URL; defaults to
                ``settings.EMBEDDING_API_URL``.
            api_key: Bearer token; defaults to ``settings.EMBEDDING_API_KEY``.
            model: Served model name; defaults to ``settings.EMBEDDING_MODEL``.
            batch_size: Passages per request; defaults to
                ``settings.EMBEDDING_BATCH_SIZE``.
        """
        settings = get_settings()
        self.model = model or settings.EMBEDDING_MODEL
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        self._client = openai.OpenAI(
            base_url=base_url or settings.EMBEDDING_API_URL,
            api_key=api_key or settings.EMBEDDING_API_KEY,
            max_retries=0,  # tenacity owns retries (see class docstring)
        )

    def embed_passages(self, texts: list[str], verbose: int | None = None) -> list[list[float]]:
        """Embed passages for indexing (e5 **passage mode** — no prefix).

        Args:
            texts: Passage texts, embedded exactly as given.
            verbose: Console verbosity (0–2); defaults to
                ``settings.DEFAULT_VERBOSE``.

        Returns:
            One embedding vector per input text, in input order.

        Raises:
            ValueError: If ``verbose`` is invalid.
            openai.APIError: If a request still fails after retries.
        """
        check_verbose(get_settings().DEFAULT_VERBOSE if verbose is None else verbose)
        for i, text in enumerate(texts):
            n_tokens = count_tokens(text)
            if n_tokens >= TOKEN_WARN_THRESHOLD:
                logger.warning(
                    "passage %d is ~%d tokens (≥%d): e5 truncates at 512 — "
                    "check CHUNK_SIZE / context blurb length",
                    i,
                    n_tokens,
                    TOKEN_WARN_THRESHOLD,
                )
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            embeddings.extend(self._embed_batch(texts[start : start + self.batch_size]))
        return embeddings

    def embed_query(self, query: str, verbose: int | None = None) -> list[float]:
        """Embed a search query (e5 **query mode** — instruction-wrapped).

        Args:
            query: The user's query; wrapped via :func:`format_query` before
                embedding.
            verbose: Console verbosity (0–2); defaults to
                ``settings.DEFAULT_VERBOSE``.

        Returns:
            The query's embedding vector.

        Raises:
            ValueError: If ``verbose`` is invalid.
            openai.APIError: If the request still fails after retries.
        """
        check_verbose(get_settings().DEFAULT_VERBOSE if verbose is None else verbose)
        return self._embed_batch([format_query(query)])[0]

    @retry(
        retry=retry_if_exception_type(_RETRYABLE_ERRORS),
        wait=wait_exponential(multiplier=0.5, max=10),
        stop=stop_after_attempt(4),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Send one embeddings request, retrying transient failures.

        Args:
            texts: The batch of already-formatted texts.

        Returns:
            One embedding per text, re-ordered to input order via the
            response's ``index`` field.
        """
        response = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
