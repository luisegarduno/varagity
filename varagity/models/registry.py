"""Model client factory (spec §12).

The reference implementation's ``util_model.py`` had a real bug here: it
branched on the *builtin* ``type`` instead of its ``model_type`` parameter,
so the ``reasoning``/``tool`` branches were dead code. This factory dispatches
on the parameter (regression-tested) and validates it.

Only ``"embedding"`` is registered in Phase 3; the LLM types (``default``,
with ``reasoning``/``tool`` aliases) land in Phase 4.
"""

from varagity.models.embeddings import EmbeddingsClient

MODEL_TYPES: tuple[str, ...] = ("embedding",)


def get_model(model_type: str) -> EmbeddingsClient:
    """Build the client for a model type.

    Args:
        model_type: One of :data:`MODEL_TYPES` (Phase 3: ``"embedding"``).

    Returns:
        A configured client for that backend.

    Raises:
        ValueError: If ``model_type`` is not a known type (message lists the
            available ones).
    """
    if model_type == "embedding":
        return EmbeddingsClient()
    raise ValueError(f"Unknown model_type {model_type!r}. Available: {list(MODEL_TYPES)}")
