"""Typed application configuration.

Settings load from the process environment and the repo-root ``.env`` file
(see ``.env.example``). Modules read the :class:`Settings` object obtained via
:func:`get_settings` — never ``os.getenv`` — so configuration stays typed,
validated, and mockable in tests.
"""

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from the environment and ``.env``.

    ``.env`` also carries lowercase Docker Compose interpolation variables
    (e.g. ``embeddings_volume``), so unknown keys are ignored rather than
    rejected.

    Defaults are the in-container values (see ``.env.example``); host runs
    override them via ``.env`` per the host-vs-container comment convention.

    Attributes:
        LOG_LEVEL: Root logging level name (e.g. ``"INFO"``, ``"DEBUG"``).
        DEFAULT_VERBOSE: Default console verbosity for public functions;
            ``0`` = off, ``1`` = low (names, counts), ``2`` = high (full
            metadata, panels).
        DOCS_PATH: Directory scanned for the ingest corpus.
        BASE_MODEL: Filename of the llama.cpp ``.gguf`` model, relative to the
            bind-mounted ``${models_volume}`` directory.
        POSTGRES_HOST: PostgreSQL host (service name in-container).
        POSTGRES_PORT: PostgreSQL port.
        POSTGRES_DB: PostgreSQL database name.
        POSTGRES_USER: PostgreSQL user.
        POSTGRES_PASSWORD: PostgreSQL password (dev-only static credential).
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    LOG_LEVEL: str = "INFO"
    DEFAULT_VERBOSE: int = 1

    DOCS_PATH: str = "./docs"
    BASE_MODEL: str = "Qwythos-9B-Claude-Mythos-5-1M-Q8_0.gguf"

    POSTGRES_HOST: str = "postgres"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "varagity"
    POSTGRES_USER: str = "varagity"
    POSTGRES_PASSWORD: str = "change-me"

    @field_validator("DEFAULT_VERBOSE")
    @classmethod
    def _validate_default_verbose(cls, value: int) -> int:
        """Reject verbosity defaults outside the supported levels.

        Args:
            value: The configured ``DEFAULT_VERBOSE`` value.

        Returns:
            The validated value, unchanged.

        Raises:
            ValueError: If ``value`` is not 0, 1, or 2.
        """
        if value not in (0, 1, 2):
            raise ValueError(f"DEFAULT_VERBOSE must be 0 (off), 1 (low), or 2 (high); got {value}")
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` instance.

    The instance is created on first call and cached. Tests that need a
    fresh load (e.g. after patching the environment) call
    ``get_settings.cache_clear()`` or construct :class:`Settings` directly.

    Returns:
        The cached application settings.
    """
    return Settings()
