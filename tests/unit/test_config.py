"""Unit tests for varagity.config (spec §15.2 "config" row)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from varagity.config import Settings, get_settings

SETTINGS_ENV_VARS = (
    "LOG_LEVEL",
    "DEFAULT_VERBOSE",
    "DOCS_PATH",
    "ALLOWED_EXTENSIONS",
    "CHUNKING_STRATEGY",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "EMBEDDING_MODEL",
    "EMBEDDING_API_URL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_DIM",
    "EMBEDDING_BATCH_SIZE",
    "BASE_MODEL",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
)


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
    assert settings.DOCS_PATH == "./docs"
    assert settings.BASE_MODEL.endswith(".gguf")
    assert settings.POSTGRES_HOST == "postgres"
    assert settings.POSTGRES_PORT == 5432
    assert settings.POSTGRES_DB == "varagity"
    assert settings.POSTGRES_USER == "varagity"
    assert settings.ALLOWED_EXTENSIONS == ".pdf,.txt,.md"
    assert settings.CHUNKING_STRATEGY == "recursive_character"
    assert settings.CHUNK_SIZE == 400  # characters, not tokens
    assert settings.CHUNK_OVERLAP == 50
    assert settings.EMBEDDING_MODEL == "infloat/multilingual-e5-large-instruct"
    assert settings.EMBEDDING_API_URL == "http://infinity-embeddings:8081/v1"
    assert settings.EMBEDDING_DIM == 1024
    assert settings.EMBEDDING_BATCH_SIZE == 32


class TestAllowedExtensionSet:
    def test_parses_and_normalizes(self) -> None:
        settings = Settings(_env_file=None, ALLOWED_EXTENSIONS=" .TXT, md ,.pdf,")
        assert settings.allowed_extension_set == frozenset({".txt", ".md", ".pdf"})

    def test_empty_whitelist_fails_fast(self) -> None:
        with pytest.raises(ValidationError, match="ALLOWED_EXTENSIONS"):
            Settings(_env_file=None, ALLOWED_EXTENSIONS=" , ,")


class TestSizeValidation:
    def test_overlap_must_be_smaller_than_chunk_size(self) -> None:
        with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
            Settings(_env_file=None, CHUNK_SIZE=100, CHUNK_OVERLAP=100)

    def test_negative_overlap_fails(self) -> None:
        with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
            Settings(_env_file=None, CHUNK_OVERLAP=-1)

    @pytest.mark.parametrize("field", ["CHUNK_SIZE", "EMBEDDING_DIM", "EMBEDDING_BATCH_SIZE"])
    def test_positive_sizes_enforced(self, field: str) -> None:
        with pytest.raises(ValidationError, match=field):
            Settings(_env_file=None, **{field: 0})


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("DEFAULT_VERBOSE", "2")
    monkeypatch.setenv("DOCS_PATH", "/somewhere/else")
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_PORT", "15432")
    settings = Settings(_env_file=None)
    assert settings.LOG_LEVEL == "DEBUG"
    assert settings.DEFAULT_VERBOSE == 2
    assert settings.DOCS_PATH == "/somewhere/else"
    assert settings.POSTGRES_HOST == "localhost"
    assert settings.POSTGRES_PORT == 15432  # coerced from str


def test_non_numeric_postgres_port_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_PORT", "not-a-port")
    with pytest.raises(ValidationError, match="POSTGRES_PORT"):
        Settings(_env_file=None)


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
