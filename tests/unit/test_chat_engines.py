"""Unit tests for the chat-engine registry and both engines (spec_v3 §4.2, §4.5).

The registry mirrors ``varagity.retrieval`` (spec §5.1): self-registration
on package import, ``KeyError``-with-listing on unknown names, and the
``@register`` decorator returning the class unchanged. ``simple`` is
today's stateless behavior verbatim — the identity split, zero LLM calls.
``condense_context`` rewrites follow-ups against history, with every
failure mode a fallback to the raw query (§4.6), never a failed turn.
"""

import dataclasses
import logging
from collections.abc import Callable, Sequence
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
from varagity.chat.condense import CondenseContextEngine
from varagity.chat.prompts import CONDENSE_PROMPT, format_history
from varagity.chat.simple import SimpleChatEngine


class ExplodingLLM:
    """Fails the test if any engine under test makes an LLM call."""

    def generate(self, messages: Any, **kwargs: Any) -> str:
        raise AssertionError("the engine must not call the LLM")

    def generate_stream(self, messages: Any, **kwargs: Any) -> Any:
        raise AssertionError("the engine must not call the LLM")


class ScriptedLLM:
    """Records generate() calls; returns a scripted response or raises."""

    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        verbose: int | None = None,
    ) -> str:
        self.calls.append({"messages": list(messages), "max_tokens": max_tokens})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class TestRegistry:
    def test_simple_is_registered_on_package_import(self) -> None:
        assert "simple" in CHAT_ENGINE_REGISTRY
        assert isinstance(CHAT_ENGINE_REGISTRY["simple"], SimpleChatEngine)

    def test_condense_context_is_registered_on_package_import(self) -> None:
        assert "condense_context" in CHAT_ENGINE_REGISTRY
        assert isinstance(CHAT_ENGINE_REGISTRY["condense_context"], CondenseContextEngine)
        assert isinstance(CHAT_ENGINE_REGISTRY["condense_context"], ChatEngine)

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


HISTORY = (
    Turn("user", "What is the Corvo Tidal Grid's most unusual feature?"),
    Turn("assistant", "Its kelp corridor. [SOURCE]: tidal_grid.txt"),
)
FOLLOW_UP = "How long is it?"
STANDALONE = "How long is the Corvo Tidal Grid's kelp corridor?"


class TestCondenseContextEngine:
    """The engine's contract: condense on history, fall back on anything else."""

    @pytest.fixture(autouse=True)
    def pinned_condense_settings(self, settings_env: Callable[..., None]) -> None:
        """Pin every knob the engine reads, so the machine's .env can't leak in."""
        settings_env(
            CONDENSE_ENABLED="true",
            CONDENSE_MODEL_TYPE="default",
            CONDENSE_HISTORY_TURNS="6",
            CONDENSE_MAX_TOKENS="128",
            CONDENSE_MAX_CHARS="512",
        )

    def prepare(self, llm: Any, history: Sequence[Turn] = HISTORY) -> PreparedQuery:
        return CondenseContextEngine().prepare(FOLLOW_UP, history=history, llm=llm, verbose=0)

    def test_condenses_the_follow_up_and_keeps_the_original(self) -> None:
        """★ The two-string split: search rewritten, original untouched."""
        llm = ScriptedLLM(STANDALONE)
        prepared = self.prepare(llm)
        assert prepared.search_query == STANDALONE
        assert prepared.original_query == FOLLOW_UP
        assert prepared.condensed is True
        assert prepared.condense_latency_s is not None
        assert prepared.condense_latency_s >= 0.0

    def test_prompt_carries_history_and_follow_up(self) -> None:
        llm = ScriptedLLM(STANDALONE)
        self.prepare(llm)
        assert len(llm.calls) == 1
        (message,) = llm.calls[0]["messages"]
        assert message["role"] == "user"
        prompt = message["content"]
        assert prompt == CONDENSE_PROMPT.format(history=format_history(HISTORY), query=FOLLOW_UP)
        assert "user: What is the Corvo Tidal Grid's most unusual feature?" in prompt
        assert "assistant: Its kelp corridor. [SOURCE]: tidal_grid.txt" in prompt
        assert f"FOLLOW-UP: {FOLLOW_UP}" in prompt

    def test_generation_is_capped_at_condense_max_tokens(self) -> None:
        llm = ScriptedLLM(STANDALONE)
        self.prepare(llm)
        assert llm.calls[0]["max_tokens"] == 128

    def test_think_block_is_stripped_before_the_search(self) -> None:
        """★ The retrieval-poisoning case (spec_v3 §4.5).

        generate() returns reasoning tags verbatim, and an unstripped
        <think> stage would be embedded as the search query — silently
        destroying retrieval.
        """
        llm = ScriptedLLM(f"<think>The user means the kelp corridor.</think>{STANDALONE}")
        prepared = self.prepare(llm)
        assert prepared.search_query == STANDALONE
        assert "<think>" not in prepared.search_query
        assert prepared.condensed is True

    def test_empty_history_makes_zero_llm_calls(self) -> None:
        """The first turn never condenses — no round-trip for nothing."""
        llm = ScriptedLLM(STANDALONE)
        prepared = self.prepare(llm, history=())
        assert llm.calls == []
        assert prepared.condensed is False
        assert prepared.search_query == FOLLOW_UP
        assert prepared.condense_latency_s is None

    def test_zero_history_turns_disables_condensing(
        self, settings_env: Callable[..., None]
    ) -> None:
        """CONDENSE_HISTORY_TURNS=0 must empty every history.

        The engine must not slice as ``[-0:]``, which would keep it all.
        """
        settings_env(CONDENSE_HISTORY_TURNS="0")
        llm = ScriptedLLM(STANDALONE)
        prepared = self.prepare(llm)
        assert llm.calls == []
        assert prepared.condensed is False

    def test_history_is_bounded_to_the_newest_turns(
        self, settings_env: Callable[..., None]
    ) -> None:
        settings_env(CONDENSE_HISTORY_TURNS="2")
        llm = ScriptedLLM(STANDALONE)
        long_history = (
            Turn("user", "ANCIENT-QUESTION"),
            Turn("assistant", "ANCIENT-ANSWER"),
            *HISTORY,
        )
        self.prepare(llm, history=long_history)
        prompt = llm.calls[0]["messages"][0]["content"]
        assert "ANCIENT-QUESTION" not in prompt
        assert "ANCIENT-ANSWER" not in prompt
        assert "kelp corridor" in prompt  # the newest pair survived

    def test_kill_switch_degrades_to_simple_and_logs(
        self, settings_env: Callable[..., None], caplog: pytest.LogCaptureFixture
    ) -> None:
        """CONDENSE_ENABLED=false is orthogonal to engine selection.

        The selected engine stays condense_context but behaves like simple.
        """
        settings_env(CONDENSE_ENABLED="false")
        llm = ScriptedLLM(STANDALONE)
        with caplog.at_level(logging.INFO, logger="varagity.chat.condense"):
            prepared = self.prepare(llm)
        assert llm.calls == []
        assert prepared == PreparedQuery(
            search_query=FOLLOW_UP,
            original_query=FOLLOW_UP,
            condensed=False,
            condense_latency_s=None,
        )
        assert "CONDENSE_ENABLED=false" in caplog.text

    def test_llm_failure_falls_back_and_warns_but_never_raises(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """§4.6: a degraded search query still answers; a 500 answers nothing."""
        llm = ScriptedLLM(RuntimeError("model server fell over"))
        with caplog.at_level(logging.WARNING, logger="varagity.chat.condense"):
            prepared = self.prepare(llm)
        assert prepared.condensed is False
        assert prepared.search_query == FOLLOW_UP
        assert "condense LLM call failed" in caplog.text

    def test_empty_result_falls_back(self, caplog: pytest.LogCaptureFixture) -> None:
        llm = ScriptedLLM("")
        with caplog.at_level(logging.WARNING, logger="varagity.chat.condense"):
            prepared = self.prepare(llm)
        assert prepared.condensed is False
        assert prepared.search_query == FOLLOW_UP
        assert "empty query" in caplog.text

    def test_think_only_response_cleans_to_empty_and_falls_back(self) -> None:
        """A reasoning stage with no query after it is an empty result."""
        llm = ScriptedLLM("<think>hmm, unclear what they mean</think>")
        prepared = self.prepare(llm)
        assert prepared.condensed is False
        assert prepared.search_query == FOLLOW_UP

    def test_overlong_result_falls_back(
        self, settings_env: Callable[..., None], caplog: pytest.LogCaptureFixture
    ) -> None:
        settings_env(CONDENSE_MAX_CHARS="32")
        llm = ScriptedLLM("x" * 33)
        with caplog.at_level(logging.WARNING, logger="varagity.chat.condense"):
            prepared = self.prepare(llm)
        assert prepared.condensed is False
        assert prepared.search_query == FOLLOW_UP
        assert "CONDENSE_MAX_CHARS" in caplog.text

    def test_none_llm_resolves_via_the_model_registry(
        self, monkeypatch: pytest.MonkeyPatch, settings_env: Callable[..., None]
    ) -> None:
        """llm=None → get_model(CONDENSE_MODEL_TYPE), like generate_answer."""
        settings_env(CONDENSE_MODEL_TYPE="reasoning")
        resolved = ScriptedLLM(STANDALONE)
        seen: list[str] = []

        def fake_get_model(model_type: str) -> ScriptedLLM:
            seen.append(model_type)
            return resolved

        monkeypatch.setattr("varagity.chat.condense.get_model", fake_get_model)
        prepared = self.prepare(llm=None)
        assert seen == ["reasoning"]
        assert len(resolved.calls) == 1
        assert prepared.search_query == STANDALONE
