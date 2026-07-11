"""Retrieval methods — one implementation per file, discovered via registry.

Importing this package imports every implementation module so each
``@register``-decorated retriever self-registers (spec §5.1). Adding a method
later (``bm25.py`` and ``hybrid.py`` in Phase 6, rerankers post-v1) means
adding the module and its import line here — no caller edits.
"""

from varagity.retrieval import semantic as _semantic  # noqa: F401  (self-registration import)
from varagity.retrieval.base import (
    RETRIEVER_REGISTRY,
    Retriever,
    get_retriever,
    register,
)

__all__ = ["RETRIEVER_REGISTRY", "Retriever", "get_retriever", "register"]
