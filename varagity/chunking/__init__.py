"""Chunking strategies — one implementation per file, discovered via registry.

Importing this package imports every implementation module so each
``@register``-decorated strategy self-registers (spec §5.1). Adding a
strategy means adding the module and its import line here — no caller
edits (proven by the v2 Phase 6 additions: ``token_based``,
``markdown_aware``, ``docling_hybrid``, ``semantic``).
"""

from varagity.chunking import docling_hybrid as _docling_hybrid  # noqa: F401
from varagity.chunking import markdown_aware as _markdown_aware  # noqa: F401
from varagity.chunking import recursive_character as _recursive_character  # noqa: F401
from varagity.chunking import semantic as _semantic  # noqa: F401
from varagity.chunking import token_based as _token_based  # noqa: F401
from varagity.chunking.base import CHUNKER_REGISTRY, ChunkingStrategy, get_chunker, register

__all__ = ["CHUNKER_REGISTRY", "ChunkingStrategy", "get_chunker", "register"]
