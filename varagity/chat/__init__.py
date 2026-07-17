"""Chat engines — one implementation per file, discovered via registry.

Importing this package imports every implementation module so each
``@register``-decorated engine self-registers (spec §5.1, spec_v3 §4.2).
Adding an engine later means adding the module and its import line here —
no caller edits, exactly as the retrieval registry's additions proved.
"""

from varagity.chat import condense as _condense  # noqa: F401  (self-registration import)
from varagity.chat import simple as _simple  # noqa: F401  (self-registration import)
from varagity.chat.base import (
    CHAT_ENGINE_REGISTRY,
    ChatEngine,
    PreparedQuery,
    Turn,
    get_chat_engine,
    register,
)

__all__ = [
    "CHAT_ENGINE_REGISTRY",
    "ChatEngine",
    "PreparedQuery",
    "Turn",
    "get_chat_engine",
    "register",
]
