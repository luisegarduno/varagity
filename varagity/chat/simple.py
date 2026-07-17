"""The ``simple`` chat engine: today's stateless behavior, verbatim (spec_v3 §4.2)."""

from collections.abc import Sequence

from varagity.chat.base import PreparedQuery, Turn, register
from varagity.models.llm import LLMClient


@register("simple")
class SimpleChatEngine:
    """Pass-through engine: search with the user's words, ignore history.

    Registering today's behavior as an engine of its own is what proves the
    registry isn't a special case built for one implementation — ``simple``
    is a peer of every engine that follows it, not the absence of one.
    """

    def prepare(
        self,
        query: str,
        *,
        history: Sequence[Turn],
        llm: LLMClient | None,
        verbose: int,
    ) -> PreparedQuery:
        """Return the identity split: the query passes through unchanged.

        Never calls a model — a ``simple`` turn costs no LLM round-trip.

        Args:
            query: The user's question, verbatim.
            history: Prior turns; deliberately ignored.
            llm: Chat client; deliberately ignored.
            verbose: Validated console verbosity; nothing to render.

        Returns:
            ``search_query == original_query``, ``condensed=False``.
        """
        return PreparedQuery(
            search_query=query,
            original_query=query,
            condensed=False,
            condense_latency_s=None,
        )
