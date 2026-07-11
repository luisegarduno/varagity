"""Document parsers — one implementation per file, discovered via registry.

Importing this package imports every implementation module so each
``@register``-decorated parser self-registers (spec §5.1). Adding a parser
means adding the module and its import line here — no caller edits.
"""

from varagity.ingest.parsers import pdf as _pdf  # noqa: F401  (self-registration import)
from varagity.ingest.parsers import text as _text  # noqa: F401  (self-registration import)
from varagity.ingest.parsers.base import (
    PARSER_REGISTRY,
    Parser,
    RawDocument,
    get_parser,
    register,
)

__all__ = ["PARSER_REGISTRY", "Parser", "RawDocument", "get_parser", "register"]
