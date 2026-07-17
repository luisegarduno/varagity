"""Chat-engine protocol and registry (spec_v3 §4.2, the spec §5.1 pattern).

Each chat-engine module defines one implementation decorated with
``@register("name")``; callers resolve the configured engine with
``get_chat_engine(settings.CHAT_ENGINE)``. Registered: ``simple`` (v3 —
today's stateless behavior, verbatim); adding an engine later means one new
file plus its import line — no caller edits, exactly as the retrieval
registry's ``reranked`` addition proved.

An engine decides **what string the retriever searches with**, given the
turn and its conversation history. It is deliberately not a
:class:`~varagity.retrieval.base.Retriever`: condensing needs chat history,
which doesn't fit ``Retriever.retrieve(query, k, …)`` — widening that
signature would force a parameter on three retrievers that will never use
it.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from varagity.models.llm import LLMClient


@dataclass(frozen=True)
class Turn:
    """One prior conversation turn, as chat engines consume it.

    Attributes:
        role: ``"user"`` or ``"assistant"``.
        content: The turn's text (the question or the generated answer).
    """

    role: str
    content: str


@dataclass(frozen=True)
class PreparedQuery:
    """A chat engine's decision about what to search with (spec_v3 §4.2).

    The two-string split is the design's core invariant: the retriever gets
    ``search_query`` while the answer prompt always gets ``original_query``
    — the user's own words and emphasis are never rewritten for generation.

    Attributes:
        search_query: What the retriever searches with (drives both the
            query embedding and BM25).
        original_query: What the answer prompt gets — always the user's
            words, verbatim.
        condensed: ``False`` means ``search_query`` is ``original_query``
            verbatim (no rewrite happened).
        condense_latency_s: Wall-clock seconds the condense LLM call took,
            or ``None`` when no call was made.
    """

    search_query: str
    original_query: str
    condensed: bool
    condense_latency_s: float | None


@runtime_checkable
class ChatEngine(Protocol):
    """Decides what string the retriever searches with, given a turn and its history.

    ``runtime_checkable`` for the same reason
    :class:`~varagity.retrieval.base.Retriever` is: the protocol appears in
    Prefect flow signatures (``varagity.pipeline.query_flow``), and Prefect
    builds a pydantic parameter schema from the annotations at decoration
    time, which requires types usable with ``isinstance``.
    """

    def prepare(
        self,
        query: str,
        *,
        history: Sequence[Turn],
        llm: LLMClient | None,
        verbose: int,
    ) -> PreparedQuery:
        """Prepare the retrieval query for one chat turn.

        Args:
            query: The user's question, verbatim.
            history: Prior turns, oldest first (empty on a first turn).
            llm: Chat client for engines that condense; ``None`` lets such
                engines resolve one via the model registry. Engines that
                never call a model ignore it.
            verbose: Validated console verbosity (0–2).

        Returns:
            The two-string split downstream stages retrieve and answer with.
        """
        ...


CHAT_ENGINE_REGISTRY: dict[str, ChatEngine] = {}


def register[T: type[Any]](name: str) -> Callable[[T], T]:
    """Class decorator registering a chat engine instance under ``name``.

    Args:
        name: Registry key (the ``CHAT_ENGINE`` env value).

    Returns:
        The decorator, which instantiates and registers the class unchanged.
    """

    def deco(cls: T) -> T:
        CHAT_ENGINE_REGISTRY[name] = cls()
        return cls

    return deco


def get_chat_engine(name: str) -> ChatEngine:
    """Look up a registered chat engine by name.

    Args:
        name: Registry key (e.g. ``"simple"``).

    Returns:
        The registered engine instance.

    Raises:
        KeyError: If no engine is registered under ``name`` (message lists
            the available ones).
    """
    if name not in CHAT_ENGINE_REGISTRY:
        raise KeyError(f"Unknown chat engine {name!r}. Available: {list(CHAT_ENGINE_REGISTRY)}")
    return CHAT_ENGINE_REGISTRY[name]
