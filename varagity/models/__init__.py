"""Self-hosted model clients (spec §12).

Both backing servers speak the OpenAI ``/v1`` surface, so the ``openai`` SDK
pointed at local base URLs is the single client library. Obtain clients via
:func:`varagity.models.registry.get_model`.
"""

from varagity.models.embeddings import EmbeddingsClient
from varagity.models.llm import LLMClient, clean_response
from varagity.models.registry import get_model

__all__ = ["EmbeddingsClient", "LLMClient", "clean_response", "get_model"]
