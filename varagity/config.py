"""Typed application configuration.

Settings load from the process environment and the repo-root ``.env`` file
(see ``.env.example``). Modules read the :class:`Settings` object obtained via
:func:`get_settings` — never ``os.getenv`` — so configuration stays typed,
validated, and mockable in tests.
"""

from functools import lru_cache

from pydantic import field_validator, model_validator
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
        ALLOWED_EXTENSIONS: Comma-separated whitelist of ingestable file
            extensions (v1: ``.pdf``, ``.txt``, ``.md``).
        CHUNKING_STRATEGY: Registry name of the chunking strategy
            (see ``varagity.chunking``).
        CHUNK_SIZE: Chunk size in **characters** — not tokens —
            (``RecursiveCharacterTextSplitter`` counts characters, spec §9.3).
        CHUNK_OVERLAP: Overlap between consecutive chunks, in characters.
        EMBEDDING_MODEL: Served model name passed to the embeddings API (the
            infinity ``INFINITY_SERVED_MODEL_NAME`` string, verbatim).
        EMBEDDING_API_URL: OpenAI-compatible base URL of the infinity server.
        EMBEDDING_API_KEY: Bearer token for the infinity server.
        EMBEDDING_DIM: Embedding dimensionality (1024 for e5-large-instruct).
        EMBEDDING_BATCH_SIZE: Number of passages sent per embeddings request.
        BASE_MODEL: Filename of the llama.cpp ``.gguf`` model, relative to the
            bind-mounted ``${models_volume}`` directory.
        BASE_MODEL_API_URL: OpenAI-compatible base URL of the llama.cpp server.
        BASE_MODEL_API_KEY: Bearer token for the llama.cpp server (it runs
            unauthenticated in v1, but the OpenAI client requires a value).
        MAX_TOKENS: Generation cap per LLM response.
        LLM_TEMPERATURE: Sampling temperature for LLM responses.
        POSTGRES_HOST: PostgreSQL host (service name in-container).
        POSTGRES_PORT: PostgreSQL port.
        POSTGRES_DB: PostgreSQL database name.
        POSTGRES_USER: PostgreSQL user.
        POSTGRES_PASSWORD: PostgreSQL password (dev-only static credential).
        RETRIEVAL_METHOD: Registry name of the retrieval method (spec §10.1:
            ``semantic`` | ``bm25`` | ``hybrid``). Defaults to ``semantic``
            until hybrid lands in Phase 6.
        TOP_K: Number of chunks retrieved per query.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    LOG_LEVEL: str = "INFO"
    DEFAULT_VERBOSE: int = 1

    DOCS_PATH: str = "./docs"
    ALLOWED_EXTENSIONS: str = ".pdf,.txt,.md"

    CHUNKING_STRATEGY: str = "recursive_character"
    CHUNK_SIZE: int = 400  # characters, not tokens (spec §9.3)
    CHUNK_OVERLAP: int = 50

    EMBEDDING_MODEL: str = "infloat/multilingual-e5-large-instruct"
    EMBEDDING_API_URL: str = "http://infinity-embeddings:8081/v1"
    EMBEDDING_API_KEY: str = "change-me"
    EMBEDDING_DIM: int = 1024
    EMBEDDING_BATCH_SIZE: int = 32

    BASE_MODEL: str = "Qwythos-9B-Claude-Mythos-5-1M-Q8_0.gguf"
    BASE_MODEL_API_URL: str = "http://llamacpp:8080/v1"
    BASE_MODEL_API_KEY: str = "none"
    MAX_TOKENS: int = 8192
    LLM_TEMPERATURE: float = 0.6

    POSTGRES_HOST: str = "postgres"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "varagity"
    POSTGRES_USER: str = "varagity"
    POSTGRES_PASSWORD: str = "change-me"

    # Temporary default — flips to "hybrid" when Phase 6 lands it (plan §4).
    RETRIEVAL_METHOD: str = "semantic"
    TOP_K: int = 10

    @property
    def allowed_extension_set(self) -> frozenset[str]:
        """Parsed ``ALLOWED_EXTENSIONS`` as a normalized set.

        Entries are lowercased, stripped, and guaranteed to start with a dot
        (``"md"`` and ``".md"`` are equivalent in the env value).

        Returns:
            The allowed extensions, e.g. ``frozenset({".pdf", ".txt", ".md"})``.
        """
        extensions = set()
        for raw in self.ALLOWED_EXTENSIONS.split(","):
            ext = raw.strip().lower()
            if not ext:
                continue
            extensions.add(ext if ext.startswith(".") else f".{ext}")
        return frozenset(extensions)

    @field_validator("ALLOWED_EXTENSIONS")
    @classmethod
    def _validate_allowed_extensions(cls, value: str) -> str:
        """Reject an extension whitelist with no usable entries.

        Args:
            value: The configured ``ALLOWED_EXTENSIONS`` value.

        Returns:
            The validated value, unchanged.

        Raises:
            ValueError: If no non-empty extension remains after splitting.
        """
        if not any(part.strip() for part in value.split(",")):
            raise ValueError("ALLOWED_EXTENSIONS must list at least one extension, e.g. '.txt,.md'")
        return value

    @model_validator(mode="after")
    def _validate_sizes(self) -> "Settings":
        """Reject size parameters that cannot produce a valid pipeline run.

        Returns:
            The validated settings instance.

        Raises:
            ValueError: If ``CHUNK_SIZE``, ``EMBEDDING_DIM``,
                ``EMBEDDING_BATCH_SIZE``, ``MAX_TOKENS``, or ``TOP_K`` is not
                positive, if ``CHUNK_OVERLAP`` is negative, or if
                ``CHUNK_OVERLAP`` is not smaller than ``CHUNK_SIZE``.
        """
        for name in ("CHUNK_SIZE", "EMBEDDING_DIM", "EMBEDDING_BATCH_SIZE", "MAX_TOKENS", "TOP_K"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive; got {getattr(self, name)}")
        if self.CHUNK_OVERLAP < 0:
            raise ValueError(f"CHUNK_OVERLAP must be non-negative; got {self.CHUNK_OVERLAP}")
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.CHUNK_OVERLAP}) must be smaller than "
                f"CHUNK_SIZE ({self.CHUNK_SIZE})"
            )
        return self

    @field_validator("RETRIEVAL_METHOD")
    @classmethod
    def _validate_retrieval_method(cls, value: str) -> str:
        """Reject retrieval methods outside the spec §10.1 vocabulary.

        Membership in the vocabulary is validated here; whether the method is
        *registered yet* (``bm25``/``hybrid`` land in Phase 6) is enforced by
        :func:`varagity.retrieval.get_retriever` at lookup time.

        Args:
            value: The configured ``RETRIEVAL_METHOD`` value.

        Returns:
            The validated value, unchanged.

        Raises:
            ValueError: If ``value`` is not ``semantic``, ``bm25``, or
                ``hybrid``.
        """
        allowed = ("semantic", "bm25", "hybrid")
        if value not in allowed:
            raise ValueError(f"RETRIEVAL_METHOD must be one of {allowed}; got {value!r}")
        return value

    @field_validator("LLM_TEMPERATURE")
    @classmethod
    def _validate_llm_temperature(cls, value: float) -> float:
        """Reject sampling temperatures outside the OpenAI-compatible range.

        Args:
            value: The configured ``LLM_TEMPERATURE`` value.

        Returns:
            The validated value, unchanged.

        Raises:
            ValueError: If ``value`` is not within ``[0.0, 2.0]``.
        """
        if not 0.0 <= value <= 2.0:
            raise ValueError(f"LLM_TEMPERATURE must be between 0.0 and 2.0; got {value}")
        return value

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
