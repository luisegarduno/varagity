"""Unit tests for the approximate token counter."""

import logging
from collections.abc import Iterator

import pytest

from varagity import tokens


@pytest.fixture
def fresh_encoder_cache() -> Iterator[None]:
    tokens._encoding.cache_clear()
    yield
    tokens._encoding.cache_clear()


def test_empty_string_is_zero_tokens() -> None:
    assert tokens.count_tokens("") == 0


def test_counts_scale_with_text() -> None:
    short = tokens.count_tokens("hello")
    longer = tokens.count_tokens("hello " * 100)
    assert 0 < short < longer
    # cl100k averages roughly one token per short word here
    assert 80 <= longer <= 220


def test_falls_back_to_char_estimate_when_encoding_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    fresh_encoder_cache: None,
) -> None:
    def boom(name: str) -> None:
        raise OSError("offline")

    monkeypatch.setattr(tokens.tiktoken, "get_encoding", boom)
    with caplog.at_level(logging.WARNING):
        estimate = tokens.count_tokens("x" * 40)
    assert estimate == 10  # len // 4
    assert any("fall back" in r.message for r in caplog.records)


def test_fallback_warning_logged_once(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    fresh_encoder_cache: None,
) -> None:
    def boom(name: str) -> None:
        raise OSError("offline")

    monkeypatch.setattr(tokens.tiktoken, "get_encoding", boom)
    with caplog.at_level(logging.WARNING):
        tokens.count_tokens("abcd")
        tokens.count_tokens("efgh")
    warnings = [r for r in caplog.records if "unavailable" in r.message]
    assert len(warnings) == 1  # lru_cache makes the failure (and warning) one-time


class TestTruncateToTokens:
    def test_text_within_budget_is_unchanged(self) -> None:
        text = "a short sentence"
        assert tokens.truncate_to_tokens(text, 100) == text

    def test_truncates_to_token_budget(self) -> None:
        text = "word " * 200
        truncated = tokens.truncate_to_tokens(text, 50)
        assert len(truncated) < len(text)
        assert tokens.count_tokens(truncated) <= 50
        assert text.startswith(truncated)  # a clean prefix, decoded back

    def test_zero_budget_returns_empty(self) -> None:
        assert tokens.truncate_to_tokens("anything", 0) == ""

    def test_negative_budget_raises(self) -> None:
        with pytest.raises(ValueError, match="max_tokens"):
            tokens.truncate_to_tokens("anything", -1)

    def test_falls_back_to_char_estimate_when_encoding_unavailable(
        self, monkeypatch: pytest.MonkeyPatch, fresh_encoder_cache: None
    ) -> None:
        def boom(name: str) -> None:
            raise OSError("offline")

        monkeypatch.setattr(tokens.tiktoken, "get_encoding", boom)
        assert tokens.truncate_to_tokens("x" * 100, 10) == "x" * 40  # 10 tokens × 4 chars
