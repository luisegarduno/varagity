"""Unit tests for the chat-engine registry and the ``simple`` engine (spec_v3 §4.2).

The registry mirrors ``varagity.retrieval`` (spec §5.1): self-registration
on package import, ``KeyError``-with-listing on unknown names, and the
``@register`` decorator returning the class unchanged. ``simple`` is
today's stateless behavior verbatim — the identity split, zero LLM calls.
"""

import dataclasses
from collections.abc import Sequence
from typing import Any

import pytest

from varagity.chat import (
    CHAT_ENGINE_REGISTRY,
    ChatEngine,
    PreparedQuery,
    Turn,
    get_chat_engine,
    register,
)
from varagity.chat.simple import SimpleChatEngine


class ExplodingLLM:
    """Fails the test if any engine under test makes an LLM call."""

    def generate(self, messages: Any, **kwargs: Any) -> str:
        raise AssertionError("the engine must not call the LLM")

    def generate_stream(self, messages: Any, **kwargs: Any) -> Any:
        raise AssertionError("the engine must not call the LLM")


class TestRegistry:
    def test_simple_is_registered_on_package_import(self) -> None:
        assert "simple" in CHAT_ENGINE_REGISTRY
        assert isinstance(CHAT_ENGINE_REGISTRY["simple"], SimpleChatEngine)

    def test_get_chat_engine_returns_the_registered_instance(self) -> None:
        assert get_chat_engine("simple") is CHAT_ENGINE_REGISTRY["simple"]

    def test_unknown_engine_raises_keyerror_listing_available(self) -> None:
        with pytest.raises(KeyError) as excinfo:
            get_chat_engine("made_up")
        message = str(excinfo.value)
        assert "made_up" in message
        assert "simple" in message  # the listing names what IS available

    def test_register_adds_an_instance_and_returns_the_class(self) -> None:
        @register("_test_probe")
        class ProbeEngine:
            def prepare(
                self,
                query: str,
                *,
                history: Sequence[Turn],
                llm: Any,
                verbose: int,
            ) -> PreparedQuery:
                return PreparedQuery(query, query, condensed=False, condense_latency_s=None)

        try:
            assert isinstance(CHAT_ENGINE_REGISTRY["_test_probe"], ProbeEngine)
            assert ProbeEngine.__name__ == "ProbeEngine"  # returned unchanged
        finally:
            del CHAT_ENGINE_REGISTRY["_test_probe"]  # keep the registry pristine

    def test_engines_satisfy_the_runtime_checkable_protocol(self) -> None:
        """Prefect's parameter-schema machinery needs isinstance to work."""
        assert isinstance(CHAT_ENGINE_REGISTRY["simple"], ChatEngine)


class TestSimpleEngine:
    def test_returns_the_query_verbatim_with_no_condense(self) -> None:
        prepared = SimpleChatEngine().prepare(
            "What powers Aurora?",
            history=(Turn("user", "earlier q"), Turn("assistant", "earlier a")),
            llm=None,
            verbose=0,
        )
        assert prepared.search_query == "What powers Aurora?"
        assert prepared.original_query == "What powers Aurora?"
        assert prepared.condensed is False
        assert prepared.condense_latency_s is None

    def test_never_calls_the_llm(self) -> None:
        """Zero model round-trips, history or not — the engine's whole contract."""
        engine = SimpleChatEngine()
        for history in ((), (Turn("user", "q0"), Turn("assistant", "a0"))):
            prepared = engine.prepare("q?", history=history, llm=ExplodingLLM(), verbose=0)  # type: ignore[arg-type]
            assert prepared.condensed is False


class TestFrozenTypes:
    def test_prepared_query_is_immutable(self) -> None:
        prepared = PreparedQuery("s", "o", condensed=False, condense_latency_s=None)
        with pytest.raises(dataclasses.FrozenInstanceError):
            prepared.search_query = "mutated"  # type: ignore[misc]

    def test_turn_is_immutable(self) -> None:
        turn = Turn("user", "q")
        with pytest.raises(dataclasses.FrozenInstanceError):
            turn.content = "mutated"  # type: ignore[misc]
