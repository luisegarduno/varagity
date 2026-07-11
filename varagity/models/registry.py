"""Model client factory (spec §12).

The reference implementation's ``util_model.py`` had a real bug here: it
branched on the *builtin* ``type`` instead of its ``model_type`` parameter,
so the ``reasoning``/``tool`` branches were dead code. This factory dispatches
on the parameter (regression-tested) and validates it.

v1 serves a single LLM, so ``reasoning`` and ``tool`` are aliases of
``default`` — one llama.cpp server hosts one model (spec §21 #9); they become
separate servers/models post-v1 without touching callers.
"""

from typing import Literal, overload

from varagity.models.embeddings import EmbeddingsClient
from varagity.models.llm import LLMClient

MODEL_TYPES: tuple[str, ...] = ("default", "embedding", "reasoning", "tool")

# LLM types that resolve to the single llama.cpp server in v1.
_LLM_TYPES: frozenset[str] = frozenset({"default", "reasoning", "tool"})


@overload
def get_model(model_type: Literal["embedding"]) -> EmbeddingsClient: ...


@overload
def get_model(model_type: Literal["default", "reasoning", "tool"]) -> LLMClient: ...


@overload
def get_model(model_type: str) -> EmbeddingsClient | LLMClient: ...


def get_model(model_type: str) -> EmbeddingsClient | LLMClient:
    """Build the client for a model type.

    Args:
        model_type: One of :data:`MODEL_TYPES`. ``embedding`` resolves to the
            infinity client; ``default`` — and its v1 aliases ``reasoning``
            and ``tool`` — to the llama.cpp chat client.

    Returns:
        A configured client for that backend.

    Raises:
        ValueError: If ``model_type`` is not a known type (message lists the
            available ones).
    """
    if model_type == "embedding":
        return EmbeddingsClient()
    if model_type in _LLM_TYPES:
        return LLMClient()
    raise ValueError(f"Unknown model_type {model_type!r}. Available: {list(MODEL_TYPES)}")
