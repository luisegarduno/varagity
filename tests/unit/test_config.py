"""Unit tests for varagity.config (spec §15.2 "config" row)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from varagity.config import Settings, get_settings

SETTINGS_ENV_VARS = ("LOG_LEVEL", "DEFAULT_VERBOSE")


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip settings env vars and reset the get_settings cache around each test."""
    for var in SETTINGS_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()


def test_defaults_load() -> None:
    settings = Settings(_env_file=None)
    assert settings.LOG_LEVEL == "INFO"
    assert settings.DEFAULT_VERBOSE == 1


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("DEFAULT_VERBOSE", "2")
    settings = Settings(_env_file=None)
    assert settings.LOG_LEVEL == "DEBUG"
    assert settings.DEFAULT_VERBOSE == 2


@pytest.mark.parametrize("bad_verbose", [-1, 3, 42])
def test_invalid_default_verbose_fails_fast(bad_verbose: int) -> None:
    with pytest.raises(ValidationError, match="DEFAULT_VERBOSE"):
        Settings(_env_file=None, DEFAULT_VERBOSE=bad_verbose)


def test_invalid_default_verbose_from_env_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEFAULT_VERBOSE", "5")
    with pytest.raises(ValidationError, match="DEFAULT_VERBOSE"):
        Settings(_env_file=None)


def test_env_file_loads_and_ignores_compose_vars(tmp_path: Path) -> None:
    """`.env` carries lowercase compose-interpolation vars; they must not reject loading."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        'embeddings_volume="/some/models/dir"\n'
        'secret_infinity_key="not-a-real-key"\n'
        "LOG_LEVEL=WARNING\n"
        "DEFAULT_VERBOSE=0\n"
    )
    settings = Settings(_env_file=env_file)
    assert settings.LOG_LEVEL == "WARNING"
    assert settings.DEFAULT_VERBOSE == 0
    assert not hasattr(settings, "embeddings_volume")


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)  # no .env here — defaults only
    first = get_settings()
    assert get_settings() is first
    get_settings.cache_clear()
    assert get_settings() is not first
