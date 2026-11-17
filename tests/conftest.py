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

    Two settings are neutralized up front to keep that promise, because a
    developer ``.env`` breaks tests that pass in CI (which has no ``.env``
    at all):

    - ``GRAPH_HANDLE_NAMES_FILE`` naming the container's contacts file makes
      :attr:`~varagity.config.Settings.graph_handle_name_map` *raise*
      wherever that path doesn't exist, failing every message-parsing test.
      The eval harness pins the same setting empty for the same reason.
    - ``RETRIEVAL_METHOD=reranked`` makes config validation reject the small
      rerank pools tests pin (``TOP_K`` must fit in ``RERANK_CANDIDATES``),
      and breaks tests asserting the default method. It is held at the
      default, ``hybrid``.

    Tests that exercise either setting pass their own value, which wins.
    """
    monkeypatch.setenv("GRAPH_HANDLE_NAMES_FILE", "")
    monkeypatch.setenv("RETRIEVAL_METHOD", "hybrid")

    def _set(**values: object) -> None:
        for name, value in values.items():
            monkeypatch.setenv(name, str(value))
        get_settings.cache_clear()

    get_settings.cache_clear()
    yield _set
    get_settings.cache_clear()
