"""infinity cross-encoder rerank client (spec_v2 §5.4).

infinity's ``/rerank`` route is **not** an OpenAI-SDK method, so this client
posts with ``httpx`` directly (same ``tenacity`` retry posture as the other
model clients). The wire contract is infinity's Cohere-protocol schema,
verified from its ``pymodels.py``: request ``{model, query, documents,
return_documents, top_n}``; response ``{"results": [{"index": i,
"relevance_score": s}, …]}`` where ``index`` is the 0-based position in the
input ``documents`` list and results arrive sorted by relevance.

Only a served **cross-encoder** (``bge-reranker-v2-m3``) can fulfill
``/rerank`` — infinity structurally rejects bi-encoders (e5, jina) with a
4xx, which this client surfaces as a clear :class:`ValueError` instead of a
bare HTTP error.
"""

import logging
from typing import Any

import httpx
from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from varagity.config import get_settings
from varagity.debug.show import check_verbose

logger = logging.getLogger(__name__)

# Cross-encoding a full RERANK_CANDIDATES pool on the batch-capped reranker
# card takes seconds, not milliseconds — well past httpx's 5 s default.
TIMEOUT_SECONDS = 120.0


def _is_retryable(error: BaseException) -> bool:
    """Whether an exception is a transient failure worth retrying.

    Same policy as the other model clients: connection/timeout trouble,
    5xx, and 429. Other 4xx (bad model, bad auth) are permanent and
    surface immediately.

    Args:
        error: The exception raised by one request attempt.

    Returns:
        ``True`` for transport errors and retryable HTTP statuses.
    """
    if isinstance(error, httpx.TransportError):
        return True
    return isinstance(error, httpx.HTTPStatusError) and (
        error.response.status_code >= 500 or error.response.status_code == 429
    )


class RerankResult(BaseModel):
    """One document's cross-encoder relevance judgment.

    Attributes:
        index: 0-based position of the document in the request's
            ``documents`` list (the Cohere protocol — results come back
            sorted by relevance, so ``index`` is the join key back to the
            input order).
        relevance_score: The cross-encoder's query–document relevance.
    """

    index: int
    relevance_score: float


class RerankClient:
    """Client for the infinity ``/rerank`` endpoint.

    Retries transient failures with exponential backoff (``tenacity`` owns
    retries, mirroring the embeddings/LLM clients). The whole candidate
    pool goes in one request — batching across the pool is the server's
    concern (infinity's ``'32;4'`` batch cap on the reranker).
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        """Create the client.

        Args:
            base_url: OpenAI-style base URL (the ``/v1`` prefix infinity
                serves under); defaults to ``settings.RERANK_API_URL``.
            api_key: Bearer token; defaults to ``settings.RERANK_API_KEY``.
            model: Served reranker name; defaults to
                ``settings.RERANK_MODEL``. Must be a cross-encoder — the
                server rejects bi-encoders at ``/rerank``.
        """
        settings = get_settings()
        self.model = model or settings.RERANK_MODEL
        self._base_url = (base_url or settings.RERANK_API_URL).rstrip("/")
        self._api_key = api_key or settings.RERANK_API_KEY

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
        verbose: int | None = None,
    ) -> list[RerankResult]:
        """Score documents against a query with the cross-encoder.

        Args:
            query: The user's query, unformatted (cross-encoders attend
                over the raw pair — no e5-style instruction wrapping).
            documents: Candidate texts to score, best-first order not
                required.
            top_n: Keep only the ``top_n`` most relevant results
                (server-side cut); ``None`` scores and returns every
                document.
            verbose: Console verbosity (0–2); defaults to
                ``settings.DEFAULT_VERBOSE``.

        Returns:
            Relevance results, sorted most-relevant first; each ``index``
            points back into ``documents``. Empty input returns ``[]``
            without a request.

        Raises:
            ValueError: If ``verbose`` is invalid, or the server rejected
                the request permanently (e.g. ``RERANK_MODEL`` is not a
                served cross-encoder).
            httpx.HTTPError: If the request still fails after retries.
        """
        check_verbose(get_settings().DEFAULT_VERBOSE if verbose is None else verbose)
        if not documents:
            return []
        payload: dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "return_documents": False,
        }
        if top_n is not None:
            payload["top_n"] = top_n
        try:
            data = self._post(payload)
        except httpx.HTTPStatusError as error:
            if _is_retryable(error):  # transient, retries exhausted — not a misconfig
                raise
            raise ValueError(
                f"rerank request rejected (HTTP {error.response.status_code}) for model "
                f"{self.model!r}: {error.response.text}. Only a served cross-encoder "
                "(e.g. BAAI/bge-reranker-v2-m3) can fulfill /rerank — bi-encoders "
                "(e5, jina) are structurally rejected."
            ) from error
        return [RerankResult.model_validate(item) for item in data["results"]]

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=0.5, max=10),
        stop=stop_after_attempt(4),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send one rerank request, retrying transient failures.

        Args:
            payload: The full request body.

        Returns:
            The parsed response body.

        Raises:
            httpx.HTTPStatusError: On a non-2xx response (retried for
                5xx/429, then reraised).
            httpx.TransportError: On connection/timeout trouble after
                retries.
        """
        response = httpx.post(
            f"{self._base_url}/rerank",
            json=payload,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result
