"""Unit tests for the contextualizer (spec §9.4, §11.1)."""

import json
import logging
import re
from collections.abc import Callable, Sequence

import httpx
import openai
import pytest
import respx

from varagity.config import Settings, get_settings
from varagity.context import contextual as contextual_module
from varagity.context.contextual import (
    _MAX_RECUTS,
    doc_token_budget,
    fit_document_preamble,
    situate_context,
)
from varagity.debug.show import console
from varagity.models.llm import LLMClient
from varagity.tokens import count_tokens, truncate_to_tokens

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

BASE_URL = "http://fake-llamacpp/v1"
CHAT = f"{BASE_URL}/chat/completions"
TOKENIZE = "http://fake-llamacpp/tokenize"  # llama.cpp's own route: the server root

# A location-history-shaped row — a timestamp and a coordinate pair, the
# shape of the Snapchat export that failed every ingest.
DIGIT_ROW = "2024-08-13 21:47:05 UTC | 37.774929, -122.419416 | ±12.5 m\n"
# What the served chat template adds around a user message (measured: the
# injected default system prompt plus role markers).
TEMPLATE_TOKENS = 91


def digit_split_count(text: str) -> int:
    """Count like the served model, faked: one token per digit, cl100k for the rest.

    cl100k packs up to three digits into a token; the served model spends
    one per digit. On :data:`DIGIT_ROW` this drifts 1.44× — the real export
    measured 1.45×.
    """
    return count_tokens(re.sub(r"\d", "", text)) + sum(char.isdigit() for char in text)


class RecordingLLM:
    """Stub LLM: records the exact messages and caps, returns a scripted response.

    ``count_served_tokens`` stands in for llama.cpp's ``/tokenize``: it counts
    like cl100k by default (no drift); ``served`` swaps in a drifting
    tokenizer, or one that cannot count (returns ``None``).
    """

    def __init__(self, response: str, served: Callable[[str], int | None] = count_tokens) -> None:
        self.response = response
        self.served = served
        self.prompts: list[str] = []
        self.roles: list[list[str]] = []
        self.max_tokens_seen: list[int | None] = []
        self.counted: list[str] = []

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

    def count_served_tokens(self, text: str) -> int | None:
        self.counted.append(text)
        return self.served(text)


@pytest.fixture
def window(settings_env: Callable[..., None]) -> int:
    """Pin the shipped window so the budget arithmetic is hermetic.

    Returns:
        The preamble budget under it (11,648 served tokens).
    """
    settings_env(LLM_CONTEXT_TOKENS=16384, CONTEXTUALIZE_MAX_TOKENS=2048, MAX_TOKENS=8192)
    return doc_token_budget(get_settings())


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


def test_preamble_is_sent_as_given(window: int) -> None:
    """situate_context never re-fits: the per-document fit already happened.

    Re-fitting per chunk would re-pay the served count per chunk, and a
    count failing mid-document would hand some chunks a different preamble
    — breaking the shared prefix llama.cpp's prompt cache relies on.
    """
    preamble = "word " * (window + 500)  # over budget on purpose: sent anyway
    llm = RecordingLLM("blurb")
    situate_context(preamble, "chunk", llm=llm, verbose=0)  # type: ignore[arg-type]
    assert llm.prompts[0].split("</document>")[0] == f"<document>\n{preamble}\n"
    assert llm.counted == []


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


class TestFitDocumentPreamble:
    """The per-document fit: cl100k first cut, the served tokenizer's last word."""

    def test_small_document_is_whole_without_a_count(
        self, window: int, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Fewer UTF-8 bytes than the budget fits any tokenizer: nothing to ask."""
        document = "word " * 100
        llm = RecordingLLM("blurb")

        def no_registry(model_type: str) -> RecordingLLM:
            raise AssertionError("a small document needs no client at all")

        monkeypatch.setattr(contextual_module, "get_model", no_registry)
        with caplog.at_level(logging.WARNING):
            assert fit_document_preamble(document, llm=llm) == document  # type: ignore[arg-type]
            assert fit_document_preamble(document) == document
        assert llm.counted == []
        assert not caplog.records

    def test_over_budget_document_is_cut_with_warning(
        self, window: int, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An over-budget document warns and is cut — never crashes ingest."""
        document = "word " * (window + 3000) + "ENDSENTINEL"
        assert count_tokens(document) > window
        llm = RecordingLLM("blurb")
        with caplog.at_level(logging.WARNING):
            preamble = fit_document_preamble(document, llm=llm)  # type: ignore[arg-type]
            situate_context(preamble, "CHUNK SENTINEL", llm=llm, verbose=0)  # type: ignore[arg-type]

        assert any("truncating the document preamble" in r.message for r in caplog.records)
        assert count_tokens(preamble) <= window
        assert llm.counted  # the served tokenizer confirmed the cut
        prompt = llm.prompts[0]
        # The document was cut (its tail is gone); the chunk and the instruction
        # scaffolding survive intact.
        assert "ENDSENTINEL" not in prompt
        assert "CHUNK SENTINEL" in prompt
        assert prompt.startswith("<document>\nword word")
        assert prompt.endswith("Answer only with the succinct context and nothing else.")

    def test_served_count_has_the_last_word_even_when_cl100k_fits(
        self, window: int, caplog: pytest.LogCaptureFixture
    ) -> None:
        """cl100k fits the whole document; the served model does not — re-cut it."""
        document = DIGIT_ROW * 300
        assert count_tokens(document) <= window < digit_split_count(document)
        llm = RecordingLLM("blurb", served=digit_split_count)
        with caplog.at_level(logging.WARNING):
            preamble = fit_document_preamble(document, llm=llm)  # type: ignore[arg-type]

        assert digit_split_count(preamble) <= window
        assert document.startswith(preamble)
        messages = [r.message for r in caplog.records]
        assert not any("truncating" in message for message in messages)  # no cl100k cut
        assert any("re-cutting" in message for message in messages)

    def test_no_served_count_falls_back_to_a_byte_safe_head(
        self, window: int, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No count (server down, not llama.cpp) → the any-tokenizer byte bound."""
        document = DIGIT_ROW * 700
        llm = RecordingLLM("blurb", served=lambda text: None)
        with caplog.at_level(logging.WARNING):
            preamble = fit_document_preamble(document, llm=llm)  # type: ignore[arg-type]

        assert len(preamble.encode("utf-8")) <= window
        assert document.startswith(preamble)
        assert len(llm.counted) == 1  # asked once, then fell back
        assert any("no served token count" in r.message for r in caplog.records)

    def test_byte_safe_head_never_splits_a_character(self, window: int) -> None:
        document = "é" * window  # two UTF-8 bytes each: twice the budget
        llm = RecordingLLM("blurb", served=lambda text: None)
        preamble = fit_document_preamble(document, llm=llm)  # type: ignore[arg-type]
        assert preamble == "é" * (window // 2)

    def test_recuts_are_bounded_then_fall_back(
        self, window: int, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A count that never lands cannot loop: bounded re-cuts, then the byte bound."""
        document = DIGIT_ROW * 700
        llm = RecordingLLM("blurb", served=lambda text: window + 1)
        with caplog.at_level(logging.WARNING):
            preamble = fit_document_preamble(document, llm=llm)  # type: ignore[arg-type]

        assert len(llm.counted) == _MAX_RECUTS + 1
        assert len(preamble.encode("utf-8")) <= window
        assert any("still" in r.message and "re-cuts" in r.message for r in caplog.records)

    def test_registry_client_is_resolved_once_a_count_is_needed(
        self, window: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        llm = RecordingLLM("blurb", served=digit_split_count)
        monkeypatch.setattr(contextual_module, "get_model", lambda model_type: llm)
        preamble = fit_document_preamble(DIGIT_ROW * 700)
        assert llm.counted
        assert digit_split_count(preamble) <= window

    def test_tiny_window_degrades_to_chunk_only_prompt(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A window too small for any preamble drops the document, not the file."""
        tiny = Settings(
            _env_file=None, LLM_CONTEXT_TOKENS=4096, CONTEXTUALIZE_MAX_TOKENS=2048, MAX_TOKENS=1024
        )
        assert doc_token_budget(tiny) <= 0
        monkeypatch.setattr(contextual_module, "get_settings", lambda: tiny)
        llm = RecordingLLM("blurb")
        with caplog.at_level(logging.WARNING):
            preamble = fit_document_preamble("DOC SENTINEL", llm=llm)  # type: ignore[arg-type]
            blurb = situate_context(preamble, "CHUNK SENTINEL", llm=llm, verbose=0)  # type: ignore[arg-type]
        assert preamble == ""
        assert blurb == "blurb"
        assert any("no room for a document preamble" in r.message for r in caplog.records)
        assert "DOC SENTINEL" not in llm.prompts[0]
        assert "CHUNK SENTINEL" in llm.prompts[0]
        assert llm.counted == []


def _served_tokenize(request: httpx.Request) -> httpx.Response:
    """A llama.cpp ``/tokenize`` that splits digits the way the served model does."""
    n_tokens = digit_split_count(json.loads(request.content)["content"])
    return httpx.Response(200, json={"tokens": list(range(n_tokens))})


def _llamacpp_chat(request: httpx.Request) -> httpx.Response:
    """A llama.cpp chat route: reject a prompt the window cannot hold, as b9935 does."""
    n_ctx = get_settings().LLM_CONTEXT_TOKENS
    prompt = json.loads(request.content)["messages"][0]["content"]
    n_prompt = digit_split_count(prompt) + TEMPLATE_TOKENS
    if n_prompt >= n_ctx:
        return httpx.Response(
            400,
            json={
                "error": {
                    "code": 400,
                    "message": f"request ({n_prompt} tokens) exceeds the available context "
                    f"size ({n_ctx} tokens), try increasing it",
                    "type": "exceed_context_size_error",
                    "n_prompt_tokens": n_prompt,
                    "n_ctx": n_ctx,
                }
            },
        )
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model.gguf",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Situated."},
                    "finish_reason": "stop",
                }
            ],
        },
    )


class TestServedWindowRegression:
    """The location_history.html failure, reproduced against a mocked llama.cpp."""

    @respx.mock
    def test_digit_dense_document_fits_the_served_window(self, window: int) -> None:
        tokenize = respx.post(TOKENIZE).mock(side_effect=_served_tokenize)
        respx.post(CHAT).mock(side_effect=_llamacpp_chat)
        llm = LLMClient(base_url=BASE_URL, api_key="none", model="test-model.gguf")
        document = DIGIT_ROW * 700

        # The undercount: cl100k says the old cut fits the budget, while the
        # served model counts it past the whole window on its own...
        old_cut = truncate_to_tokens(document, window)
        assert count_tokens(old_cut) <= window
        assert digit_split_count(old_cut) > get_settings().LLM_CONTEXT_TOKENS
        # ...so llama.cpp refused every chunk, as it did the real file.
        with pytest.raises(openai.BadRequestError, match="exceed_context_size_error"):
            situate_context(old_cut, DIGIT_ROW, llm=llm, verbose=0)
        assert tokenize.call_count == 0  # the old path never asked the server

        preamble = fit_document_preamble(document, llm=llm)

        assert digit_split_count(preamble) <= window
        assert document.startswith(preamble)
        assert situate_context(preamble, DIGIT_ROW, llm=llm, verbose=0) == "Situated."
        # One count of the cl100k cut, one re-cut, one count confirming it.
        assert tokenize.call_count == 2


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
