"""Model client factory (spec §12).

The reference implementation's ``util_model.py`` had a real bug here: it
branched on the *builtin* ``type`` instead of its ``model_type`` parameter,
so the ``reasoning``/``tool`` branches were dead code. This factory dispatches
on the parameter (regression-tested) and validates it.

v1 serves a single LLM, so ``reasoning`` and ``tool`` are aliases of
``default`` — one llama.cpp server hosts one model (spec §21 #9); they become
separate servers/models post-v1 without touching callers. v2 adds ``rerank``
(spec_v2 §5.4): the infinity-served cross-encoder.
"""

from typing import Literal, overload

from varagity.models.embeddings import EmbeddingsClient
from varagity.models.llm import LLMClient
from varagity.models.rerank import RerankClient

MODEL_TYPES: tuple[str, ...] = ("default", "embedding", "rerank", "reasoning", "tool")

# LLM aliases that resolve to the single llama.cpp server in v1/v2 — the
# valid CHAT_MODEL_TYPE vocabulary (spec_v2 §4.7), exposed by GET /api/config
# so the GUI's model-type controls offer exactly these.
LLM_MODEL_TYPES: tuple[str, ...] = ("default", "reasoning", "tool")

_LLM_TYPES: frozenset[str] = frozenset(LLM_MODEL_TYPES)


@overload
def get_model(model_type: Literal["embedding"]) -> EmbeddingsClient: ...


@overload
def get_model(model_type: Literal["rerank"]) -> RerankClient: ...


@overload
def get_model(model_type: Literal["default", "reasoning", "tool"]) -> LLMClient: ...


@overload
def get_model(model_type: str) -> EmbeddingsClient | LLMClient | RerankClient: ...


def get_model(model_type: str) -> EmbeddingsClient | LLMClient | RerankClient:
    """Build the client for a model type.

    Args:
        model_type: One of :data:`MODEL_TYPES`. ``embedding`` resolves to the
            infinity embeddings client; ``rerank`` to the infinity
            cross-encoder client (spec_v2 §5.4); ``default`` — and its v1
            aliases ``reasoning`` and ``tool`` — to the llama.cpp chat
            client.

    Returns:
        A configured client for that backend.

    Raises:
        ValueError: If ``model_type`` is not a known type (message lists the
            available ones).
    """
    if model_type == "embedding":
        return EmbeddingsClient()
    if model_type == "rerank":
        return RerankClient()
    if model_type in _LLM_TYPES:
        return LLMClient()
    raise ValueError(f"Unknown model_type {model_type!r}. Available: {list(MODEL_TYPES)}")
