"""Unit tests for the contextualizer (spec §9.4, §11.1)."""

import logging
from collections.abc import Sequence

import pytest

from varagity.config import Settings, get_settings
from varagity.context import contextual as contextual_module
from varagity.context.contextual import doc_token_budget, situate_context
from varagity.debug.show import console
from varagity.tokens import count_tokens

# The spec §11.1 cookbook prompt, written out literally so the test fails if
# the template drifts by even a character (plan: "prompt formatted exactly").
EXPECTED_PROMPT = """<document>
DOC TEXT
</document>

Here is the chunk we want to situate within the whole document
<chunk>
CHUNK TEXT
</chunk>

Please give a short succinct context to situate this chunk within the overall
document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else."""


class RecordingLLM:
    """Stub LLM: records the exact messages and caps, returns a scripted response."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []
        self.roles: list[list[str]] = []
        self.max_tokens_seen: list[int | None] = []

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        verbose: int | None = None,
    ) -> str:
        self.roles.append([message["role"] for message in messages])
        self.prompts.append(messages[0]["content"])
        self.max_tokens_seen.append(max_tokens)
        return self.response


def test_prompt_is_formatted_exactly() -> None:
    """Placeholder substitution and nothing else — verbatim spec §11.1."""
    llm = RecordingLLM("The blurb.")
    blurb = situate_context("DOC TEXT", "CHUNK TEXT", llm=llm, verbose=0)  # type: ignore[arg-type]
    assert blurb == "The blurb."
    assert llm.prompts == [EXPECTED_PROMPT]
    assert llm.roles == [["user"]]  # one single-turn user message


def test_think_block_is_stripped() -> None:
    llm = RecordingLLM("<think>the chunk mentions turbines…</think>Describes the tidal arrays.")
    blurb = situate_context("doc", "chunk", llm=llm, verbose=0)  # type: ignore[arg-type]
    assert blurb == "Describes the tidal arrays."


def test_blurb_is_whitespace_stripped() -> None:
    llm = RecordingLLM("\n  Positions the chunk in the report.  \n")
    blurb = situate_context("doc", "chunk", llm=llm, verbose=0)  # type: ignore[arg-type]
    assert blurb == "Positions the chunk in the report."


def test_long_document_truncated_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    """An over-budget document warns and truncates — never crashes ingest."""
    budget = doc_token_budget(get_settings())
    document = "word " * (budget + 3000) + "ENDSENTINEL"
    assert count_tokens(document) > budget
    llm = RecordingLLM("blurb")
    with caplog.at_level(logging.WARNING):
        situate_context(document, "CHUNK SENTINEL", llm=llm, verbose=0)  # type: ignore[arg-type]

    assert any("truncating the document preamble" in r.message for r in caplog.records)
    prompt = llm.prompts[0]
    # The document was cut (its tail is gone); the chunk and the instruction
    # scaffolding survive intact.
    assert "ENDSENTINEL" not in prompt
    assert "CHUNK SENTINEL" in prompt
    assert prompt.startswith("<document>\nword word")
    assert prompt.endswith("Answer only with the succinct context and nothing else.")
    doc_section = prompt.split("</document>")[0]
    assert count_tokens(doc_section) <= budget + 10  # scaffolding slack


def test_document_within_budget_is_not_truncated(caplog: pytest.LogCaptureFixture) -> None:
    document = "word " * 100
    llm = RecordingLLM("blurb")
    with caplog.at_level(logging.WARNING):
        situate_context(document, "chunk", llm=llm, verbose=0)  # type: ignore[arg-type]
    assert not any("truncating" in r.message for r in caplog.records)
    assert document in llm.prompts[0]


def test_blurb_generation_uses_the_contextualize_cap() -> None:
    """Blurbs run under CONTEXTUALIZE_MAX_TOKENS, never the chat-sized cap.

    The regression that motivated this: inheriting MAX_TOKENS=8192 reserves
    half a 16k window for generation, so llama.cpp hard-rejects any document
    over ~8k tokens ("Context size has been exceeded") and the file fails
    ingest.
    """
    llm = RecordingLLM("blurb")
    situate_context("doc", "chunk", llm=llm, verbose=0)  # type: ignore[arg-type]
    assert llm.max_tokens_seen == [get_settings().CONTEXTUALIZE_MAX_TOKENS]


def test_doc_token_budget_reserves_generation_chunk_and_scaffolding() -> None:
    """The preamble budget is the window minus every reserve — never more."""
    settings = get_settings()
    budget = doc_token_budget(settings)
    assert 0 < budget < settings.LLM_CONTEXT_TOKENS - settings.CONTEXTUALIZE_MAX_TOKENS
    # Regression bound: a preamble at the budget plus the reserves must fit
    # the window, so a maximal prompt can never trip llama.cpp's hard 500.
    assert budget + settings.CONTEXTUALIZE_MAX_TOKENS < settings.LLM_CONTEXT_TOKENS


def test_tiny_window_degrades_to_chunk_only_prompt(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A window too small for any preamble drops the document, not the file."""
    tiny = Settings(
        _env_file=None, LLM_CONTEXT_TOKENS=4096, CONTEXTUALIZE_MAX_TOKENS=2048, MAX_TOKENS=1024
    )
    assert doc_token_budget(tiny) <= 0
    monkeypatch.setattr(contextual_module, "get_settings", lambda: tiny)
    llm = RecordingLLM("blurb")
    with caplog.at_level(logging.WARNING):
        blurb = situate_context("DOC SENTINEL", "CHUNK SENTINEL", llm=llm, verbose=0)  # type: ignore[arg-type]
    assert blurb == "blurb"
    assert any("no room for a document preamble" in r.message for r in caplog.records)
    assert "DOC SENTINEL" not in llm.prompts[0]
    assert "CHUNK SENTINEL" in llm.prompts[0]


def test_empty_blurb_warns_instead_of_raising(caplog: pytest.LogCaptureFixture) -> None:
    """An unclosed <think> (token cap mid-reasoning) cleans to nothing."""
    llm = RecordingLLM("<think>ran out of tok")
    with caplog.at_level(logging.WARNING):
        blurb = situate_context("doc", "chunk", llm=llm, verbose=0)  # type: ignore[arg-type]
    assert blurb == ""
    assert any("empty blurb" in r.message for r in caplog.records)


def test_resolves_llm_from_registry_when_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = RecordingLLM("registry blurb")
    monkeypatch.setattr(contextual_module, "get_model", lambda model_type: llm)
    assert situate_context("doc", "chunk", verbose=0) == "registry blurb"
    assert len(llm.prompts) == 1


def test_invalid_verbose_raises() -> None:
    with pytest.raises(ValueError, match="verbose"):
        situate_context("doc", "chunk", llm=RecordingLLM("x"), verbose=7)  # type: ignore[arg-type]


class TestRendering:
    def test_v2_renders_blurb_panel(self) -> None:
        llm = RecordingLLM("Sits within the maintenance chapter.")
        with console.capture() as capture:
            situate_context("doc", "a chunk about winch motors", llm=llm, verbose=2)  # type: ignore[arg-type]
        out = capture.get()
        assert "context blurb" in out
        assert "Sits within the maintenance chapter." in out
        assert "winch motors" in out

    @pytest.mark.parametrize("verbose", [0, 1])
    def test_low_verbosity_renders_nothing(self, verbose: int) -> None:
        llm = RecordingLLM("quiet blurb")
        with console.capture() as capture:
            situate_context("doc", "chunk", llm=llm, verbose=verbose)  # type: ignore[arg-type]
        assert capture.get() == ""
