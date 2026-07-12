"""Retrieval methods — one implementation per file, discovered via registry.

Importing this package imports every implementation module so each
``@register``-decorated retriever self-registers (spec §5.1). Adding a
method later means adding the module and its import line here — no caller
edits, as the v1 ``bm25``/``hybrid`` and v2 ``reranked`` additions proved.
"""

from varagity.retrieval import bm25 as _bm25  # noqa: F401  (self-registration import)
from varagity.retrieval import hybrid as _hybrid  # noqa: F401  (self-registration import)
from varagity.retrieval import reranked as _reranked  # noqa: F401  (self-registration import)
from varagity.retrieval import semantic as _semantic  # noqa: F401  (self-registration import)
from varagity.retrieval.base import (
    RETRIEVER_REGISTRY,
    Retriever,
    get_retriever,
    register,
)

__all__ = ["RETRIEVER_REGISTRY", "Retriever", "get_retriever", "register"]
