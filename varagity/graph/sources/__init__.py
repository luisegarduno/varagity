"""Message sources — one implementation per file, discovered via registry.

Importing this package imports every implementation module so each
``@register``-decorated source self-registers (the spec §5.1 pattern, applied
to the message-source family — spec_graphrag §5.2). Adding a platform later
means adding the module and its import line here; selection is structural
(:func:`~varagity.graph.sources.base.find_message_source`), so there is no
caller edit and no ``config.py`` vocabulary tuple to update.
"""

from varagity.graph.sources import imessage as _imessage  # noqa: F401  (self-registration import)
from varagity.graph.sources.base import (
    MESSAGE_SOURCE_REGISTRY,
    MessageBatch,
    MessageSource,
    SourceMessage,
    Tapback,
    batch_for_path,
    find_message_source,
    get_message_source,
    register,
)

__all__ = [
    "MESSAGE_SOURCE_REGISTRY",
    "MessageBatch",
    "MessageSource",
    "SourceMessage",
    "Tapback",
    "batch_for_path",
    "find_message_source",
    "get_message_source",
    "register",
]
