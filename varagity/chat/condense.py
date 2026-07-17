"""The ``condense_context`` chat engine (spec_v3 §4.5, §4.6).

Rewrites a follow-up question into a standalone search query against the
conversation history — one non-streaming LLM call before retrieval — while
the answer prompt still gets the user's original words (the
:class:`~varagity.chat.base.PreparedQuery` two-string split).

Failure is a fallback, not an error (§4.6): a transient LLM failure (after
the client's own ``tenacity`` retries), an empty result, or an absurdly
long one all degrade to searching with the raw query at ``WARNING`` — a
degraded search query still answers a lot of questions; a 500 answers
none. ``CONDENSE_ENABLED=false`` is the kill switch, checked inside
:meth:`~CondenseContextEngine.prepare` exactly as ``RERANK_ENABLED`` is
checked inside the ``reranked`` retriever — deliberately orthogonal to
engine selection.
"""

import logging
import time
from collections.abc import Sequence
from typing import cast

from varagity.chat.base import PreparedQuery, Turn, register
from varagity.chat.prompts import CONDENSE_PROMPT, format_history
from varagity.config import get_settings
from varagity.debug.show import v_condensed
from varagity.models.llm import LLMClient, clean_response
from varagity.models.registry import get_model

logger = logging.getLogger(__name__)


@register("condense_context")
class CondenseContextEngine:
    """Condense + Context: search with a history-resolved standalone query.

    The LlamaIndex "Condense Plus Context" pattern adapted to the registry
    convention: the condensed rewrite drives retrieval (both the query
    embedding and BM25) while generation always receives the original
    question — the user's own words and emphasis are never rewritten for
    the answer prompt.
    """

    def prepare(
        self,
        query: str,
        *,
        history: Sequence[Turn],
        llm: LLMClient | None,
        verbose: int,
    ) -> PreparedQuery:
        """Condense the turn against its history into the retrieval query.

        The first turn never condenses: empty history means there is
        nothing to resolve a reference against, and the common single-turn
        question must not pay an LLM round-trip for nothing. The history
        fed to the prompt is bounded by ``CONDENSE_HISTORY_TURNS`` (newest
        turns kept), whatever the caller loaded.

        Args:
            query: The user's question, verbatim.
            history: Prior turns, oldest first (empty on a first turn).
            llm: Chat client; resolved via the model registry
                (``settings.CONDENSE_MODEL_TYPE``) when ``None``.
            verbose: Validated console verbosity (0–2).

        Returns:
            The condensed split, or the identity split whenever no LLM
            call was made (kill switch, first turn) or the call's outcome
            was unusable (the §4.6 fallback).
        """
        settings = get_settings()
        identity = PreparedQuery(
            search_query=query, original_query=query, condensed=False, condense_latency_s=None
        )
        if not settings.CONDENSE_ENABLED:
            logger.info(
                "CONDENSE_ENABLED=false — degrading to the 'simple' engine's identity split"
            )
            return identity
        turns = self._bounded(history, settings.CONDENSE_HISTORY_TURNS)
        if not turns:
            return identity

        prompt = CONDENSE_PROMPT.format(history=format_history(turns), query=query)
        # CONDENSE_MODEL_TYPE is validated to the LLM aliases, so the
        # registry always resolves an LLMClient here; the cast (rather than
        # an isinstance gate) keeps duck-typed test doubles usable at this
        # seam, matching the flows' injectable-fake convention.
        client = (
            llm if llm is not None else cast("LLMClient", get_model(settings.CONDENSE_MODEL_TYPE))
        )
        started = time.perf_counter()
        try:
            # verbose=0: the sub-call renders nothing; v_condensed below is
            # this stage's console output (the reranked-retriever pattern).
            raw = client.generate(
                [{"role": "user", "content": prompt}],
                max_tokens=settings.CONDENSE_MAX_TOKENS,
                verbose=0,
            )
        except Exception:  # any failure falls back — the turn must not die here (§4.6)
            logger.warning("condense LLM call failed — searching with the raw query", exc_info=True)
            return identity
        latency_s = time.perf_counter() - started

        # Mandatory: generate() returns reasoning tags verbatim, and an
        # unstripped <think> block would go straight into the embedding
        # model — the single easiest way to silently destroy retrieval
        # quality in this feature (spec_v3 §4.5).
        condensed = clean_response(raw)
        if not condensed:
            logger.warning("condenser returned an empty query — searching with the raw query")
            return identity
        if len(condensed) > settings.CONDENSE_MAX_CHARS:
            logger.warning(
                "condensed query is %d chars (CONDENSE_MAX_CHARS=%d) — the condenser "
                "misbehaved; searching with the raw query",
                len(condensed),
                settings.CONDENSE_MAX_CHARS,
            )
            return identity
        v_condensed(query, condensed, verbose)
        return PreparedQuery(
            search_query=condensed,
            original_query=query,
            condensed=True,
            condense_latency_s=latency_s,
        )

    @staticmethod
    def _bounded(history: Sequence[Turn], limit: int) -> Sequence[Turn]:
        """Keep the newest ``limit`` turns (order preserved).

        API callers already bound their history load in SQL with the same
        setting; this bound covers history-accumulating callers like the
        CLI, whose in-memory list grows for the whole session.

        Args:
            history: Prior turns, oldest first.
            limit: Maximum turns to keep; ``0`` keeps none (a negative
                slice bound would wrap, so it is handled explicitly).

        Returns:
            The newest ``limit`` turns, oldest first.
        """
        if limit <= 0:
            return ()
        return history[-limit:]
