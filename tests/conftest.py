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
    """

    def _set(**values: object) -> None:
        for name, value in values.items():
            monkeypatch.setenv(name, str(value))
        get_settings.cache_clear()

    get_settings.cache_clear()
    yield _set
    get_settings.cache_clear()
