"""Shared fixtures for all test layers."""

from collections.abc import Callable, Iterator

import pytest

from varagity.config import get_settings


@pytest.fixture
def settings_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., None]]:
    """Override settings via env vars and reset the settings cache.

    Environment variables take precedence over the repo-root ``.env`` in
    pydantic-settings, so this gives tests hermetic control even on a
    machine whose ``.env`` exists. The cache is cleared on entry and exit.

    One setting is neutralized up front to keep that promise: a developer
    ``.env`` naming a ``GRAPH_HANDLE_NAMES_FILE`` — the container's contacts
    file, say — makes :attr:`~varagity.config.Settings.graph_handle_name_map`
    *raise* wherever that path doesn't exist, so every message-parsing test
    would fail on that machine and pass in CI (which has no ``.env`` at
    all). The eval harness pins the same setting empty for the same reason.
    Tests that exercise the file pass their own value, which wins.
    """
    monkeypatch.setenv("GRAPH_HANDLE_NAMES_FILE", "")

    def _set(**values: object) -> None:
        for name, value in values.items():
            monkeypatch.setenv(name, str(value))
        get_settings.cache_clear()

    get_settings.cache_clear()
    yield _set
    get_settings.cache_clear()
