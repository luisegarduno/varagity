"""Unit tests for varagity.logging_setup."""

import logging
from collections.abc import Iterator

import pytest
from rich.logging import RichHandler

from varagity.logging_setup import _NOISY_LOGGERS, setup_logging


@pytest.fixture(autouse=True)
def _restore_root_logger() -> Iterator[None]:
    """Snapshot and restore global logging state around each test."""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    saved_noisy_levels = {name: logging.getLogger(name).level for name in _NOISY_LOGGERS}
    yield
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)
    for name, level in saved_noisy_levels.items():
        logging.getLogger(name).setLevel(level)


def test_installs_rich_handler_at_requested_level() -> None:
    setup_logging("DEBUG")
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert any(isinstance(h, RichHandler) for h in root.handlers)


def test_level_is_case_insensitive() -> None:
    setup_logging("warning")
    assert logging.getLogger().level == logging.WARNING


def test_repeated_setup_does_not_stack_handlers() -> None:
    setup_logging("INFO")
    handler_count = len(logging.getLogger().handlers)
    setup_logging("INFO")
    assert len(logging.getLogger().handlers) == handler_count


def test_noisy_third_party_loggers_are_quieted() -> None:
    setup_logging("DEBUG")
    for name in _NOISY_LOGGERS:
        assert logging.getLogger(name).level == logging.WARNING


def test_unknown_level_raises() -> None:
    with pytest.raises(ValueError, match="Unknown level"):
        setup_logging("NOT_A_LEVEL")
