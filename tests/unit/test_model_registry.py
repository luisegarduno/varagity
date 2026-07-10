"""Unit tests for the model factory (spec §15.2 "models/registry" row)."""

import pytest

from varagity.models import get_model
from varagity.models.embeddings import EmbeddingsClient
from varagity.models.registry import MODEL_TYPES


def test_embedding_returns_embeddings_client() -> None:
    client = get_model("embedding")
    assert isinstance(client, EmbeddingsClient)


def test_dispatch_reacts_to_the_parameter() -> None:
    """Dispatch must depend on the ``model_type`` argument value.

    Regression guard for the reference bug: ``util_model.py`` branched on the
    *builtin* ``type`` instead of its ``model_type`` argument, so non-default
    branches were dead code.
    """
    assert isinstance(get_model("embedding"), EmbeddingsClient)
    with pytest.raises(ValueError, match="Unknown model_type"):
        get_model("definitely-not-a-model-type")


@pytest.mark.parametrize("bad_type", ["", "bogus", "EMBEDDING", "embeddings"])
def test_unknown_type_raises_listing_available(bad_type: str) -> None:
    with pytest.raises(ValueError, match="embedding"):
        get_model(bad_type)


def test_phase_3_registers_only_embedding() -> None:
    assert MODEL_TYPES == ("embedding",)
