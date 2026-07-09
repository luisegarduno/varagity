"""Central logging configuration.

Installs a :class:`rich.logging.RichHandler` on the root logger, once, at
application startup. Modules obtain their own logger with
``logging.getLogger(__name__)`` and never configure handlers themselves —
this module is the only place logging is configured (spec §14, channel 2).
"""

import logging

from rich.logging import RichHandler

# Third-party loggers that are noisy at INFO and below; pinned to WARNING so
# application logs stay readable at LOG_LEVEL=INFO/DEBUG.
_NOISY_LOGGERS: tuple[str, ...] = ("httpx", "httpcore", "urllib3")


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging with rich console output.

    Safe to call more than once: the root handler set is replaced on each
    call, so repeated setup (tests, re-entrant CLIs) never stacks duplicate
    handlers.

    Args:
        level: Root log level name, e.g. ``"DEBUG"`` or ``"INFO"``
            (case-insensitive).

    Raises:
        ValueError: If ``level`` is not a recognized logging level name.
    """
    logging.basicConfig(
        level=level.upper(),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        force=True,
    )
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
