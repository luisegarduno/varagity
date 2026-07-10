"""Chunking strategies — one implementation per file, discovered via registry.

Importing this package imports every implementation module so each
``@register``-decorated strategy self-registers (spec §5.1). Adding a
strategy later (semantic, markdown-aware, token-based, …) means adding the
module and its import line here — no caller edits.
"""

from varagity.chunking import recursive_character as _recursive_character  # noqa: F401
from varagity.chunking.base import CHUNKER_REGISTRY, ChunkingStrategy, get_chunker, register

__all__ = ["CHUNKER_REGISTRY", "ChunkingStrategy", "get_chunker", "register"]
