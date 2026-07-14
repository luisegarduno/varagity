"""Unit tests for the runtime settings override layer (spec_v2 §4.7).

The layer's contract: overrides export as env vars (merging over ``.env``
exactly like ``pinned_eval_settings``), an invalid merge rolls back
atomically, clearing an override restores the pre-override environment,
and the startup replay drops rows that no longer validate instead of
crashing the API.
"""

from collections.abc import Iterator

import pytest
from pydantic import ValidationError

from varagity.api import runtime_settings
from varagity.api.runtime_settings import (
    OVERRIDABLE,
    REINGEST_AFFECTING,
    active_overrides,
    apply_overrides,
    load_persisted_overrides,
    to_env_value,
)
from varagity.config import Settings, get_settings


@pytest.fixture(autouse=True)
def isolate_overrides() -> Iterator[None]:
    """Reset the module's process-global override state around every test."""
    runtime_settings.reset_for_tests()
    yield
    runtime_settings.reset_for_tests()


class TestCatalog:
    def test_every_entry_is_a_real_settings_field(self) -> None:
        fields = set(Settings.model_fields)
        assert set(OVERRIDABLE) <= fields

    def test_groups_are_the_spec_47_drawer_groups(self) -> None:
        assert {spec.group for spec in OVERRIDABLE.values()} == {
            "retrieval",
            "generation",
            "ingestion",
        }

    def test_reingest_affecting_is_the_ingest_time_knob_set(self) -> None:
        """Exactly the spec §4.7 ingest-time knobs — the v1 footgun set."""
        assert {
            "CHUNKING_STRATEGY",
            "CHUNK_SIZE",
            "CHUNK_OVERLAP",
            "CONTEXTUALIZE",
            "OCR_ENGINE",
        } == REINGEST_AFFECTING

    def test_choices_come_from_the_registries(self) -> None:
        """A new registry file must appear in the drawer automatically."""
        import varagity.chunking  # noqa: F401 — trigger self-registration
        import varagity.retrieval  # noqa: F401

        choices = {
            name: spec.choices() for name, spec in OVERRIDABLE.items() if spec.choices is not None
        }
        assert "reranked" in choices["RETRIEVAL_METHOD"]
        assert {"markdown_aware", "recursive_character", "token_based"} <= set(
            choices["CHUNKING_STRATEGY"]
        )
        assert choices["RERANK_BASE_METHOD"] == ["semantic", "bm25", "hybrid"]
        assert choices["CHAT_MODEL_TYPE"] == ["default", "reasoning", "tool"]
        assert "easyocr" in choices["OCR_ENGINE"]


class TestToEnvValue:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(True, "true"), (False, "false"), (25, "25"), (0.7, "0.7"), ("hybrid", "hybrid")],
    )
    def test_scalars_render_in_dotenv_form(self, value: object, expected: str) -> None:
        assert to_env_value(value) == expected  # type: ignore[arg-type]


class TestApplyOverrides:
    def test_override_takes_effect_and_clearing_restores(self) -> None:
        base_top_k = get_settings().TOP_K
        settings = apply_overrides({"TOP_K": str(base_top_k + 13)})
        assert base_top_k + 13 == settings.TOP_K
        assert base_top_k + 13 == get_settings().TOP_K
        assert active_overrides() == {"TOP_K": str(base_top_k + 13)}

        apply_overrides({})
        assert base_top_k == get_settings().TOP_K
        assert active_overrides() == {}

    def test_pre_existing_env_var_is_restored_not_deleted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import os

        monkeypatch.setenv("TOP_K", "7")
        get_settings.cache_clear()
        assert get_settings().TOP_K == 7

        apply_overrides({"TOP_K": "9"})
        assert get_settings().TOP_K == 9
        apply_overrides({})
        assert os.environ["TOP_K"] == "7"
        assert get_settings().TOP_K == 7

    def test_invalid_merge_rolls_back_atomically(self) -> None:
        base_top_k = get_settings().TOP_K
        apply_overrides({"TOP_K": str(base_top_k + 1)})
        with pytest.raises(ValidationError):
            apply_overrides({"TOP_K": "0"})  # positive-only validator
        assert base_top_k + 1 == get_settings().TOP_K  # previous override intact
        assert active_overrides() == {"TOP_K": str(base_top_k + 1)}

    def test_cross_field_validators_run_on_the_merged_whole(self) -> None:
        """One weight alone breaks the sum-to-1 pair; both together pass."""
        with pytest.raises(ValidationError):
            apply_overrides({"SEMANTIC_WEIGHT": "0.6"})
        settings = apply_overrides({"SEMANTIC_WEIGHT": "0.6", "BM25_WEIGHT": "0.4"})
        assert settings.SEMANTIC_WEIGHT == 0.6
        assert settings.BM25_WEIGHT == 0.4

    def test_unknown_key_is_rejected(self) -> None:
        with pytest.raises(KeyError, match="POSTGRES_PASSWORD"):
            apply_overrides({"POSTGRES_PASSWORD": "nope"})


class TestLoadPersistedOverrides:
    def test_valid_rows_apply_on_startup(self) -> None:
        base_top_k = get_settings().TOP_K
        load_persisted_overrides(lambda: {"TOP_K": str(base_top_k + 5)})
        assert base_top_k + 5 == get_settings().TOP_K

    def test_invalid_rows_boot_on_env_defaults(self, caplog: pytest.LogCaptureFixture) -> None:
        base_top_k = get_settings().TOP_K
        with caplog.at_level("ERROR"):
            load_persisted_overrides(lambda: {"TOP_K": "0"})
        assert base_top_k == get_settings().TOP_K
        assert active_overrides() == {}
        assert "no longer validate" in caplog.text

    def test_non_overridable_rows_are_dropped_with_a_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        base_top_k = get_settings().TOP_K
        with caplog.at_level("ERROR"):
            load_persisted_overrides(lambda: {"NOT_A_SETTING": "x", "TOP_K": str(base_top_k + 3)})
        assert base_top_k + 3 == get_settings().TOP_K  # the valid row still applies
        assert "NOT_A_SETTING" in caplog.text
