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
    "CONTEXTUALIZE",
    "EMBEDDING_MODEL",
    "EMBEDDING_API_URL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_DIM",
    "EMBEDDING_BATCH_SIZE",
    "BASE_MODEL",
    "BASE_MODEL_API_URL",
    "BASE_MODEL_API_KEY",
    "MAX_TOKENS",
    "LLM_TEMPERATURE",
    "LLM_CONTEXT_TOKENS",
    "CONTEXTUALIZE_MAX_TOKENS",
    "CHAT_MODEL_TYPE",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "ELASTICSEARCH_URL",
    "BM25_INDEX_NAME",
    "PREVIEW_ENABLED",
    "PREVIEW_RENDER_WIDTH",
    "PREVIEW_MIN_COVERAGE",
    "PREVIEW_CONVERT_TIMEOUT_S",
    "RETRIEVAL_METHOD",
    "TOP_K",
    "SEMANTIC_WEIGHT",
    "BM25_WEIGHT",
    "RERANK_ENABLED",
    "RERANK_MODEL",
    "RERANK_API_URL",
    "RERANK_API_KEY",
    "RERANK_TOP_N",
    "RERANK_BASE_METHOD",
    "RERANK_CANDIDATES",
    "API_HOST",
    "API_PORT",
    "API_CORS_ORIGINS",
    "UPLOAD_MAX_MB",
    "UPLOAD_MAX_FILES",
    "UPLOAD_MAX_TOTAL_MB",
    "UPLOAD_MAX_PATH_DEPTH",
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
    assert settings.ALLOWED_EXTENSIONS == ".pdf,.txt,.md,.docx,.pptx,.xlsx,.html,.htm"
    assert settings.CHUNKING_STRATEGY == "recursive_character"
    assert settings.CHUNK_SIZE == 400  # characters, not tokens
    assert settings.CHUNK_OVERLAP == 50
    assert settings.CONTEXTUALIZE is True  # Contextual Retrieval on by default
    assert settings.EMBEDDING_MODEL == "infloat/multilingual-e5-large-instruct"
    assert settings.EMBEDDING_API_URL == "http://infinity-embeddings:8081/v1"
    assert settings.EMBEDDING_DIM == 1024
    assert settings.EMBEDDING_BATCH_SIZE == 32
    assert settings.BASE_MODEL_API_URL == "http://llamacpp:8080/v1"
    assert settings.BASE_MODEL_API_KEY == "none"
    assert settings.MAX_TOKENS == 8192
    assert settings.LLM_TEMPERATURE == 0.6
    assert settings.LLM_CONTEXT_TOKENS == 16384  # mirrors the compose --ctx-size
    assert settings.CONTEXTUALIZE_MAX_TOKENS == 2048
    assert settings.ELASTICSEARCH_URL == "http://elasticsearch:9200"
    assert settings.BM25_INDEX_NAME == "varagity_contextual_bm25"
    assert settings.RETRIEVAL_METHOD == "hybrid"  # the v1 default (spec §10.1)
    assert settings.TOP_K == 10
    assert settings.SEMANTIC_WEIGHT == 0.8
    assert settings.BM25_WEIGHT == 0.2
    assert settings.RERANK_ENABLED is False  # kill switch off by default (spec_v2 §5)
    assert settings.RERANK_MODEL == "BAAI/bge-reranker-v2-m3"
    assert settings.RERANK_API_URL == "http://infinity-embeddings:8081/v1"
    assert settings.RERANK_TOP_N == 5
    assert settings.RERANK_BASE_METHOD == "hybrid"
    assert settings.RERANK_CANDIDATES == 40
    assert settings.PREVIEW_ENABLED is True  # page preview on by default (ADR-010)
    assert settings.PREVIEW_RENDER_WIDTH == 1536
    assert settings.PREVIEW_MIN_COVERAGE == 0.3
    assert settings.PREVIEW_CONVERT_TIMEOUT_S == 120


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

    @pytest.mark.parametrize(
        "field",
        [
            "CHUNK_SIZE",
            "EMBEDDING_DIM",
            "EMBEDDING_BATCH_SIZE",
            "MAX_TOKENS",
            "TOP_K",
            "LLM_CONTEXT_TOKENS",
            "CONTEXTUALIZE_MAX_TOKENS",
        ],
    )
    def test_positive_sizes_enforced(self, field: str) -> None:
        with pytest.raises(ValidationError, match=field):
            Settings(_env_file=None, **{field: 0})

    @pytest.mark.parametrize("field", ["MAX_TOKENS", "CONTEXTUALIZE_MAX_TOKENS"])
    def test_generation_caps_must_fit_the_context_window(self, field: str) -> None:
        """A cap at or over the window leaves no room for any prompt."""
        kwargs = {"LLM_CONTEXT_TOKENS": 4096, "MAX_TOKENS": 1024, field: 4096}
        with pytest.raises(ValidationError, match=field):
            Settings(_env_file=None, **kwargs)

    def test_generation_cap_under_the_window_is_valid(self) -> None:
        settings = Settings(_env_file=None, LLM_CONTEXT_TOKENS=4096, MAX_TOKENS=4095)
        assert settings.MAX_TOKENS == 4095


class TestRetrievalMethodValidation:
    @pytest.mark.parametrize("method", ["semantic", "bm25", "hybrid", "reranked"])
    def test_spec_vocabulary_accepted(self, method: str) -> None:
        """All spec §10.1 + spec_v2 §5 values pass config validation."""
        settings = Settings(_env_file=None, RETRIEVAL_METHOD=method)
        assert method == settings.RETRIEVAL_METHOD

    @pytest.mark.parametrize("bad", ["keyword", "SEMANTIC", "", "hybrid ", "rerank"])
    def test_unknown_method_fails_fast(self, bad: str) -> None:
        with pytest.raises(ValidationError, match="RETRIEVAL_METHOD"):
            Settings(_env_file=None, RETRIEVAL_METHOD=bad)


class TestChatModelTypeValidation:
    @pytest.mark.parametrize("alias", ["default", "reasoning", "tool"])
    def test_llm_aliases_accepted(self, alias: str) -> None:
        settings = Settings(_env_file=None, CHAT_MODEL_TYPE=alias)
        assert alias == settings.CHAT_MODEL_TYPE

    @pytest.mark.parametrize("bad", ["embedding", "rerank", "DEFAULT", "", "chat"])
    def test_non_chat_types_fail_fast(self, bad: str) -> None:
        """embedding/rerank are model types but not chat models."""
        with pytest.raises(ValidationError, match="CHAT_MODEL_TYPE"):
            Settings(_env_file=None, CHAT_MODEL_TYPE=bad)

    def test_vocabulary_matches_the_registry_aliases(self) -> None:
        """config.py hard-codes the tuple (circular import); keep them equal."""
        from varagity.models.registry import LLM_MODEL_TYPES

        for alias in LLM_MODEL_TYPES:
            assert alias == Settings(_env_file=None, CHAT_MODEL_TYPE=alias).CHAT_MODEL_TYPE
        assert len(LLM_MODEL_TYPES) == 3


class TestRerankValidation:
    """Rerank narrows a wider pool — the spec_v2 §5.3 cross-constraints."""

    def test_defaults_are_valid(self) -> None:
        settings = Settings(_env_file=None)
        assert settings.RERANK_TOP_N <= settings.RERANK_CANDIDATES

    @pytest.mark.parametrize("top_n", [0, -1])
    def test_nonpositive_top_n_fails_fast(self, top_n: int) -> None:
        with pytest.raises(ValidationError, match="RERANK_TOP_N"):
            Settings(_env_file=None, RERANK_TOP_N=top_n)

    def test_top_n_beyond_candidates_fails_fast(self) -> None:
        with pytest.raises(ValidationError, match="RERANK_CANDIDATES"):
            Settings(_env_file=None, RERANK_TOP_N=50, RERANK_CANDIDATES=40)

    @pytest.mark.parametrize("base", ["semantic", "bm25", "hybrid"])
    def test_base_method_vocabulary_accepted(self, base: str) -> None:
        settings = Settings(_env_file=None, RERANK_BASE_METHOD=base)
        assert base == settings.RERANK_BASE_METHOD

    @pytest.mark.parametrize("bad", ["reranked", "keyword", ""])
    def test_bad_base_method_fails_fast(self, bad: str) -> None:
        """`reranked` as its own base would recurse — rejected outright."""
        with pytest.raises(ValidationError, match="RERANK_BASE_METHOD"):
            Settings(_env_file=None, RERANK_BASE_METHOD=bad)

    def test_reranked_method_caps_top_n_at_top_k(self) -> None:
        with pytest.raises(ValidationError, match="TOP_K"):
            Settings(_env_file=None, RETRIEVAL_METHOD="reranked", RERANK_TOP_N=15, TOP_K=10)

    def test_reranked_method_needs_pool_at_least_top_k(self) -> None:
        with pytest.raises(ValidationError, match="RERANK_CANDIDATES"):
            Settings(
                _env_file=None,
                RETRIEVAL_METHOD="reranked",
                TOP_K=50,
                RERANK_TOP_N=5,
                RERANK_CANDIDATES=40,
            )

    def test_other_methods_ignore_the_top_k_coupling(self) -> None:
        """A hybrid config with unused rerank staging must not be rejected."""
        settings = Settings(
            _env_file=None, RETRIEVAL_METHOD="hybrid", TOP_K=50, RERANK_CANDIDATES=40
        )
        assert settings.TOP_K == 50

    def test_valid_reranked_config_accepted(self) -> None:
        settings = Settings(
            _env_file=None,
            RETRIEVAL_METHOD="reranked",
            TOP_K=10,
            RERANK_TOP_N=5,
            RERANK_CANDIDATES=40,
        )
        assert settings.RETRIEVAL_METHOD == "reranked"


class TestFusionWeightValidation:
    """Hybrid rank-fusion weights must form a convex blend (spec §6, §15.2)."""

    @pytest.mark.parametrize(
        ("semantic", "bm25"),
        [(0.8, 0.2), (0.5, 0.5), (1.0, 0.0), (0.0, 1.0)],
    )
    def test_weights_summing_to_one_accepted(self, semantic: float, bm25: float) -> None:
        settings = Settings(_env_file=None, SEMANTIC_WEIGHT=semantic, BM25_WEIGHT=bm25)
        assert semantic == settings.SEMANTIC_WEIGHT
        assert bm25 == settings.BM25_WEIGHT

    def test_binary_float_sum_tolerated(self) -> None:
        """0.7 + 0.3 != 1.0 exactly in binary floats — must still pass."""
        settings = Settings(_env_file=None, SEMANTIC_WEIGHT=0.7, BM25_WEIGHT=0.3)
        assert settings.SEMANTIC_WEIGHT == 0.7

    @pytest.mark.parametrize(("semantic", "bm25"), [(0.8, 0.3), (0.5, 0.4), (1.0, 1.0)])
    def test_sum_not_one_fails_fast(self, semantic: float, bm25: float) -> None:
        with pytest.raises(ValidationError, match="must sum to 1.0"):
            Settings(_env_file=None, SEMANTIC_WEIGHT=semantic, BM25_WEIGHT=bm25)

    def test_negative_weight_fails_fast(self) -> None:
        with pytest.raises(ValidationError, match="non-negative"):
            Settings(_env_file=None, SEMANTIC_WEIGHT=1.2, BM25_WEIGHT=-0.2)


class TestLLMTemperatureValidation:
    @pytest.mark.parametrize("temperature", [0.0, 0.6, 2.0])
    def test_in_range_accepted(self, temperature: float) -> None:
        settings = Settings(_env_file=None, LLM_TEMPERATURE=temperature)
        assert temperature == settings.LLM_TEMPERATURE

    @pytest.mark.parametrize("temperature", [-0.1, 2.1, 100.0])
    def test_out_of_range_fails_fast(self, temperature: float) -> None:
        with pytest.raises(ValidationError, match="LLM_TEMPERATURE"):
            Settings(_env_file=None, LLM_TEMPERATURE=temperature)


class TestPreviewValidation:
    @pytest.mark.parametrize("width", [512, 1536, 4096])
    def test_render_width_range_accepted(self, width: int) -> None:
        assert width == Settings(_env_file=None, PREVIEW_RENDER_WIDTH=width).PREVIEW_RENDER_WIDTH

    @pytest.mark.parametrize("width", [0, 511, 4097])
    def test_render_width_out_of_range_fails_fast(self, width: int) -> None:
        with pytest.raises(ValidationError, match="PREVIEW_RENDER_WIDTH"):
            Settings(_env_file=None, PREVIEW_RENDER_WIDTH=width)

    @pytest.mark.parametrize("coverage", [0.0, 0.3, 1.0])
    def test_min_coverage_range_accepted(self, coverage: float) -> None:
        settings = Settings(_env_file=None, PREVIEW_MIN_COVERAGE=coverage)
        assert coverage == settings.PREVIEW_MIN_COVERAGE

    @pytest.mark.parametrize("coverage", [-0.1, 1.1])
    def test_min_coverage_out_of_range_fails_fast(self, coverage: float) -> None:
        with pytest.raises(ValidationError, match="PREVIEW_MIN_COVERAGE"):
            Settings(_env_file=None, PREVIEW_MIN_COVERAGE=coverage)

    @pytest.mark.parametrize("timeout", [0, -5])
    def test_nonpositive_convert_timeout_fails_fast(self, timeout: int) -> None:
        with pytest.raises(ValidationError, match="PREVIEW_CONVERT_TIMEOUT_S"):
            Settings(_env_file=None, PREVIEW_CONVERT_TIMEOUT_S=timeout)


class TestContextualize:
    @pytest.mark.parametrize(("raw", "expected"), [("false", False), ("true", True)])
    def test_bool_env_parse(
        self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool
    ) -> None:
        monkeypatch.setenv("CONTEXTUALIZE", raw)
        assert Settings(_env_file=None).CONTEXTUALIZE is expected

    def test_non_bool_fails_fast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CONTEXTUALIZE", "maybe")
        with pytest.raises(ValidationError, match="CONTEXTUALIZE"):
            Settings(_env_file=None)


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


class TestApiSettings:
    def test_defaults(self) -> None:
        settings = Settings(_env_file=None)
        assert settings.API_HOST == "0.0.0.0"
        assert settings.API_PORT == 8000
        assert settings.UPLOAD_MAX_MB == 50
        assert settings.cors_origin_list == ["http://localhost:3000"]

    @pytest.mark.parametrize("bad_port", [0, -1, 65536])
    def test_out_of_range_api_port_fails_fast(self, bad_port: int) -> None:
        with pytest.raises(ValidationError, match="API_PORT"):
            Settings(_env_file=None, API_PORT=bad_port)

    @pytest.mark.parametrize("bad_mb", [0, -5])
    def test_non_positive_upload_cap_fails_fast(self, bad_mb: int) -> None:
        with pytest.raises(ValidationError, match="UPLOAD_MAX_MB"):
            Settings(_env_file=None, UPLOAD_MAX_MB=bad_mb)

    def test_upload_batch_defaults(self) -> None:
        settings = Settings(_env_file=None)
        assert settings.UPLOAD_MAX_FILES == 500
        assert settings.UPLOAD_MAX_TOTAL_MB == 2048
        assert settings.UPLOAD_MAX_PATH_DEPTH == 12

    @pytest.mark.parametrize(
        "field", ["UPLOAD_MAX_FILES", "UPLOAD_MAX_TOTAL_MB", "UPLOAD_MAX_PATH_DEPTH"]
    )
    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_batch_caps_fail_fast(self, field: str, bad: int) -> None:
        with pytest.raises(ValidationError, match=field):
            Settings(_env_file=None, **{field: bad})

    def test_per_file_cap_above_batch_budget_fails_fast(self) -> None:
        """A per-file cap larger than the whole batch's budget is a config bug."""
        with pytest.raises(ValidationError, match="UPLOAD_MAX_TOTAL_MB"):
            Settings(_env_file=None, UPLOAD_MAX_MB=100, UPLOAD_MAX_TOTAL_MB=50)

    def test_per_file_cap_equal_to_batch_budget_is_valid(self) -> None:
        settings = Settings(_env_file=None, UPLOAD_MAX_MB=50, UPLOAD_MAX_TOTAL_MB=50)
        assert settings.UPLOAD_MAX_TOTAL_MB == 50

    def test_cors_origins_parse_strip_and_dedupe(self) -> None:
        settings = Settings(
            _env_file=None,
            API_CORS_ORIGINS=" http://localhost:3000/ ,https://varagity.local, http://localhost:3000",
        )
        assert settings.cors_origin_list == [
            "http://localhost:3000",
            "https://varagity.local",
        ]
