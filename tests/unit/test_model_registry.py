"""Unit tests for the model factory (spec §15.2 "models/registry" row)."""

import pytest

from varagity.models import get_model
from varagity.models.embeddings import EmbeddingsClient
from varagity.models.llm import LLMClient
from varagity.models.registry import MODEL_TYPES


def test_embedding_returns_embeddings_client() -> None:
    client = get_model("embedding")
    assert isinstance(client, EmbeddingsClient)


def test_default_returns_llm_client() -> None:
    client = get_model("default")
    assert isinstance(client, LLMClient)


@pytest.mark.parametrize("alias", ["reasoning", "tool"])
def test_reasoning_and_tool_alias_the_default_server(alias: str) -> None:
    """v1 serves one LLM; the aliases resolve to the same client (spec §21 #9)."""
    client = get_model(alias)
    assert isinstance(client, LLMClient)


def test_dispatch_reacts_to_the_parameter() -> None:
    """Dispatch must depend on the ``model_type`` argument value.

    Regression guard for the reference bug: ``util_model.py`` branched on the
    *builtin* ``type`` instead of its ``model_type`` argument, so non-default
    branches were dead code. Each distinct argument value must reach its own
    branch.
    """
    assert isinstance(get_model("embedding"), EmbeddingsClient)
    assert isinstance(get_model("default"), LLMClient)
    assert isinstance(get_model("reasoning"), LLMClient)
    assert isinstance(get_model("tool"), LLMClient)
    with pytest.raises(ValueError, match="Unknown model_type"):
        get_model("definitely-not-a-model-type")


@pytest.mark.parametrize("bad_type", ["", "bogus", "EMBEDDING", "embeddings", "DEFAULT"])
def test_unknown_type_raises_listing_available(bad_type: str) -> None:
    with pytest.raises(ValueError, match="embedding"):
        get_model(bad_type)


def test_phase_4_registers_llm_and_embedding_types() -> None:
    assert MODEL_TYPES == ("default", "embedding", "reasoning", "tool")
