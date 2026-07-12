"""Unit tests for the chunking registry and every registered strategy."""

import logging
from collections.abc import Callable

import pytest

from varagity.chunking import CHUNKER_REGISTRY, get_chunker
from varagity.chunking.base import warn_near_token_ceiling
from varagity.chunking.markdown_aware import MarkdownAwareStrategy
from varagity.chunking.recursive_character import RecursiveCharacterStrategy
from varagity.chunking.semantic import SemanticStrategy
from varagity.chunking.token_based import TokenBasedStrategy

SOURCE_META = {
    "source": "/abs/corpus/a.md",
    "file_name": "a.md",
    "file_type": "md",
    "page": None,
}


@pytest.fixture
def small_chunks(settings_env: Callable[..., None]) -> None:
    settings_env(CHUNK_SIZE=100, CHUNK_OVERLAP=20)


class TestRegistry:
    def test_every_strategy_registered(self) -> None:
        assert isinstance(get_chunker("recursive_character"), RecursiveCharacterStrategy)
        assert sorted(CHUNKER_REGISTRY) == [
            "docling_hybrid",
            "markdown_aware",
            "recursive_character",
            "semantic",
            "token_based",
        ]

    def test_unknown_name_raises_listing_available(self) -> None:
        with pytest.raises(KeyError, match="recursive_character"):
            get_chunker("sentence_window")


class TestRecursiveCharacterStrategy:
    def test_short_text_is_one_chunk(self, small_chunks: None) -> None:
        chunks = RecursiveCharacterStrategy().split("tiny", source_meta=SOURCE_META, verbose=0)
        assert len(chunks) == 1
        assert chunks[0].page_content == "tiny"

    def test_chunk_size_respected(self, small_chunks: None) -> None:
        text = "word " * 200  # ~1000 chars with plenty of split points
        chunks = RecursiveCharacterStrategy().split(text, source_meta=SOURCE_META, verbose=0)
        assert len(chunks) > 1
        assert all(len(c.page_content) <= 100 for c in chunks)

    def test_overlap_on_unbreakable_text(self, small_chunks: None) -> None:
        """Separator-free text splits into fixed windows with exact overlap."""
        text = "x" * 250
        chunks = RecursiveCharacterStrategy().split(text, source_meta=SOURCE_META, verbose=0)
        assert [len(c.page_content) for c in chunks] == [100, 100, 90]
        for left, right in zip(chunks, chunks[1:], strict=False):
            assert left.page_content[-20:] == right.page_content[:20]

    def test_metadata_seeded_and_chunk_indexed(self, small_chunks: None) -> None:
        text = "para one." + "\n\n" + "para two " * 30
        chunks = RecursiveCharacterStrategy().split(text, source_meta=SOURCE_META, verbose=0)
        for index, chunk in enumerate(chunks):
            assert chunk.metadata["source"] == SOURCE_META["source"]
            assert chunk.metadata["file_name"] == "a.md"
            assert chunk.metadata["file_type"] == "md"
            assert chunk.metadata["page"] is None
            assert chunk.metadata["chunk_index"] == index

    def test_metadata_not_shared_between_chunks(self, small_chunks: None) -> None:
        text = "word " * 200
        chunks = RecursiveCharacterStrategy().split(text, source_meta=SOURCE_META, verbose=0)
        chunks[0].metadata["context"] = "chunk-0-only"
        assert "context" not in chunks[1].metadata

    def test_deterministic(self, small_chunks: None) -> None:
        text = ("Sentence with several words. " * 40).strip()
        first = RecursiveCharacterStrategy().split(text, source_meta=SOURCE_META, verbose=0)
        second = RecursiveCharacterStrategy().split(text, source_meta=SOURCE_META, verbose=0)
        assert [c.page_content for c in first] == [c.page_content for c in second]
        assert [c.metadata for c in first] == [c.metadata for c in second]

    def test_invalid_verbose_raises(self, small_chunks: None) -> None:
        with pytest.raises(ValueError, match="verbose"):
            RecursiveCharacterStrategy().split("text", source_meta=SOURCE_META, verbose=9)


def _words(n: int) -> str:
    return " ".join(f"word{i}" for i in range(n))


class TestTokenBasedStrategy:
    def test_fake_tokenizer_respects_token_budget(self, settings_env: Callable[..., None]) -> None:
        """CHUNK_SIZE counts tokens (here: whitespace words), not characters."""
        settings_env(CHUNK_SIZE=10, CHUNK_OVERLAP=2)
        strategy = TokenBasedStrategy(length_function=lambda text: len(text.split()))
        chunks = strategy.split(_words(53), source_meta=SOURCE_META, verbose=0)
        assert len(chunks) > 1
        assert all(len(c.page_content.split()) <= 10 for c in chunks)
        # Character lengths far exceed 10 — the budget is not characters.
        assert all(len(c.page_content) > 10 for c in chunks)

    def test_default_tiktoken_splitter_respects_token_budget(
        self, settings_env: Callable[..., None]
    ) -> None:
        from varagity.tokens import count_tokens

        settings_env(CHUNK_SIZE=100, CHUNK_OVERLAP=10)
        chunks = TokenBasedStrategy().split(_words(400), source_meta=SOURCE_META, verbose=0)
        assert len(chunks) > 1
        # +2 slack: langchain sizes by summing piece lengths, and re-tokenizing
        # the joined text can differ by a token at the seams (module docstring).
        assert all(count_tokens(c.page_content) <= 102 for c in chunks)

    def test_metadata_seeded_and_chunk_indexed(self, settings_env: Callable[..., None]) -> None:
        settings_env(CHUNK_SIZE=10, CHUNK_OVERLAP=2)
        strategy = TokenBasedStrategy(length_function=lambda text: len(text.split()))
        chunks = strategy.split(_words(30), source_meta=SOURCE_META, verbose=0)
        for index, chunk in enumerate(chunks):
            assert chunk.metadata["source"] == SOURCE_META["source"]
            assert chunk.metadata["file_type"] == "md"
            assert chunk.metadata["chunk_index"] == index
        chunks[0].metadata["context"] = "chunk-0-only"
        assert "context" not in chunks[1].metadata

    def test_deterministic(self, settings_env: Callable[..., None]) -> None:
        settings_env(CHUNK_SIZE=10, CHUNK_OVERLAP=2)
        strategy = TokenBasedStrategy(length_function=lambda text: len(text.split()))
        text = _words(40)
        first = strategy.split(text, source_meta=SOURCE_META, verbose=0)
        second = strategy.split(text, source_meta=SOURCE_META, verbose=0)
        assert [c.page_content for c in first] == [c.page_content for c in second]

    def test_near_ceiling_chunk_warns(
        self, settings_env: Callable[..., None], caplog: pytest.LogCaptureFixture
    ) -> None:
        settings_env(CHUNK_SIZE=600, CHUNK_OVERLAP=10)
        strategy = TokenBasedStrategy(length_function=lambda text: len(text.split()))
        with caplog.at_level(logging.WARNING, logger="varagity.chunking.base"):
            strategy.split(_words(520), source_meta=SOURCE_META, verbose=0)
        assert any("e5 truncates" in record.message for record in caplog.records)

    def test_invalid_verbose_raises(self, small_chunks: None) -> None:
        with pytest.raises(ValueError, match="verbose"):
            TokenBasedStrategy().split("text", source_meta=SOURCE_META, verbose=3)


MARKDOWN = """Preamble before any heading.

# Operations

Intro paragraph under the top heading.

## Dredging

The dredger Moorhen cleared the channel to nine meters.

### Channel depth

Detail line about depth.

## Berthing

```
# not a heading, just code
```

The ferry switches to battery power near the dock.
"""


class TestMarkdownAwareStrategy:
    def test_splits_on_headings_and_carries_heading_path(self, small_chunks: None) -> None:
        chunks = MarkdownAwareStrategy().split(MARKDOWN, source_meta=SOURCE_META, verbose=0)
        by_path = {c.metadata.get("heading_path"): c.page_content for c in chunks}
        assert None in by_path  # the preamble has no heading
        assert "Preamble" in by_path[None]
        assert "Intro paragraph" in by_path["Operations"]
        assert "nine meters" in by_path["Operations > Dredging"]
        assert "Detail line" in by_path["Operations > Dredging > Channel depth"]
        assert "battery power" in by_path["Operations > Berthing"]

    def test_section_keeps_its_own_heading_line(self, small_chunks: None) -> None:
        chunks = MarkdownAwareStrategy().split(MARKDOWN, source_meta=SOURCE_META, verbose=0)
        dredging = next(
            c for c in chunks if c.metadata.get("heading_path") == "Operations > Dredging"
        )
        assert dredging.page_content.startswith("## Dredging")

    def test_heading_inside_code_fence_is_content(self, small_chunks: None) -> None:
        chunks = MarkdownAwareStrategy().split(MARKDOWN, source_meta=SOURCE_META, verbose=0)
        assert not any("not a heading" in (c.metadata.get("heading_path") or "") for c in chunks)
        berthing = next(
            c for c in chunks if c.metadata.get("heading_path") == "Operations > Berthing"
        )
        assert "# not a heading, just code" in berthing.page_content

    def test_sibling_heading_resets_deeper_levels(self, small_chunks: None) -> None:
        text = "# A\n\nbody a\n\n## B\n\nbody b\n\n## C\n\nbody c\n"
        chunks = MarkdownAwareStrategy().split(text, source_meta=SOURCE_META, verbose=0)
        paths = [c.metadata.get("heading_path") for c in chunks]
        assert paths == ["A", "A > B", "A > C"]

    def test_oversized_section_resplit_inherits_path(
        self, settings_env: Callable[..., None]
    ) -> None:
        settings_env(CHUNK_SIZE=80, CHUNK_OVERLAP=10)
        text = "# Long\n\n" + ("sentence with several words. " * 20).strip() + "\n"
        chunks = MarkdownAwareStrategy().split(text, source_meta=SOURCE_META, verbose=0)
        assert len(chunks) > 1
        assert all(c.metadata["heading_path"] == "Long" for c in chunks)
        assert all(len(c.page_content) <= 80 for c in chunks)

    def test_heading_only_section_is_dropped_but_titles_survive_in_paths(
        self, small_chunks: None
    ) -> None:
        text = "# Top\n\n## Sub\n\nonly the subsection has body text\n"
        chunks = MarkdownAwareStrategy().split(text, source_meta=SOURCE_META, verbose=0)
        assert [c.metadata.get("heading_path") for c in chunks] == ["Top > Sub"]

    def test_chunk_index_sequential_and_metadata_not_shared(self, small_chunks: None) -> None:
        chunks = MarkdownAwareStrategy().split(MARKDOWN, source_meta=SOURCE_META, verbose=0)
        assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
        chunks[0].metadata["context"] = "chunk-0-only"
        assert "context" not in chunks[1].metadata

    def test_deterministic(self, small_chunks: None) -> None:
        first = MarkdownAwareStrategy().split(MARKDOWN, source_meta=SOURCE_META, verbose=0)
        second = MarkdownAwareStrategy().split(MARKDOWN, source_meta=SOURCE_META, verbose=0)
        assert [c.page_content for c in first] == [c.page_content for c in second]
        assert [c.metadata for c in first] == [c.metadata for c in second]

    def test_plain_text_without_headings_degrades_gracefully(self, small_chunks: None) -> None:
        text = "Just a paragraph. " * 3
        chunks = MarkdownAwareStrategy().split(text, source_meta=SOURCE_META, verbose=0)
        assert chunks
        assert all("heading_path" not in c.metadata for c in chunks)

    def test_invalid_verbose_raises(self, small_chunks: None) -> None:
        with pytest.raises(ValueError, match="verbose"):
            MarkdownAwareStrategy().split("text", source_meta=SOURCE_META, verbose=-1)


class FakeTopicEmbeddings:
    """Deterministic embeddings: axis selected by a topic keyword."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_passages(self, texts: list[str], verbose: int | None = None) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[0.0, 1.0] if "beta" in text.lower() else [1.0, 0.0] for text in texts]


ALPHA_BETA = (
    "Alpha reactors hum quietly. Alpha crews rotate monthly. Alpha filters need cleaning. "
    "Beta gardens grow kelp. Beta divers plant the fronds. Beta harvests peak in spring."
)


class TestSemanticStrategy:
    def test_splits_at_the_topic_shift(self, small_chunks: None) -> None:
        fake = FakeTopicEmbeddings()
        chunks = SemanticStrategy(embeddings=fake).split(  # type: ignore[arg-type]
            ALPHA_BETA, source_meta=SOURCE_META, verbose=0
        )
        assert len(chunks) == 2
        assert "beta" not in chunks[0].page_content.lower()
        assert "Beta harvests peak in spring." in chunks[1].page_content
        assert len(fake.calls) == 1  # one batched passage request

    def test_chunks_are_verbatim_substrings(self, small_chunks: None) -> None:
        chunks = SemanticStrategy(embeddings=FakeTopicEmbeddings()).split(  # type: ignore[arg-type]
            ALPHA_BETA, source_meta=SOURCE_META, verbose=0
        )
        assert all(c.page_content in ALPHA_BETA for c in chunks)

    def test_deterministic(self, small_chunks: None) -> None:
        first = SemanticStrategy(embeddings=FakeTopicEmbeddings()).split(  # type: ignore[arg-type]
            ALPHA_BETA, source_meta=SOURCE_META, verbose=0
        )
        second = SemanticStrategy(embeddings=FakeTopicEmbeddings()).split(  # type: ignore[arg-type]
            ALPHA_BETA, source_meta=SOURCE_META, verbose=0
        )
        assert [c.page_content for c in first] == [c.page_content for c in second]

    def test_short_text_is_one_chunk_without_an_embeddings_call(self, small_chunks: None) -> None:
        fake = FakeTopicEmbeddings()
        chunks = SemanticStrategy(embeddings=fake).split(  # type: ignore[arg-type]
            "One sentence only.", source_meta=SOURCE_META, verbose=0
        )
        assert len(chunks) == 1
        assert chunks[0].page_content == "One sentence only."
        assert fake.calls == []

    def test_uniform_similarity_group_is_token_resplit(
        self, settings_env: Callable[..., None]
    ) -> None:
        """All-equal distances mean no semantic breaks; the ceiling still holds."""
        from varagity.tokens import count_tokens

        settings_env(CHUNK_SIZE=30, CHUNK_OVERLAP=5)
        text = "Alpha pumps cycle seawater. " * 40
        chunks = SemanticStrategy(embeddings=FakeTopicEmbeddings()).split(  # type: ignore[arg-type]
            text, source_meta=SOURCE_META, verbose=0
        )
        assert len(chunks) > 1
        # +2 slack for the tokenizer seam nuance (see token_based's docstring).
        assert all(count_tokens(c.page_content) <= 32 for c in chunks)

    def test_resolves_client_from_model_registry_when_not_injected(
        self, small_chunks: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeTopicEmbeddings()
        monkeypatch.setattr("varagity.chunking.semantic.get_model", lambda model_type: fake)
        chunks = SemanticStrategy().split(ALPHA_BETA, source_meta=SOURCE_META, verbose=0)
        assert len(chunks) == 2
        assert fake.calls  # the registry-resolved client was used

    def test_metadata_seeded_and_chunk_indexed(self, small_chunks: None) -> None:
        chunks = SemanticStrategy(embeddings=FakeTopicEmbeddings()).split(  # type: ignore[arg-type]
            ALPHA_BETA, source_meta=SOURCE_META, verbose=0
        )
        for index, chunk in enumerate(chunks):
            assert chunk.metadata["source"] == SOURCE_META["source"]
            assert chunk.metadata["chunk_index"] == index
        chunks[0].metadata["context"] = "chunk-0-only"
        assert "context" not in chunks[1].metadata

    def test_empty_text_yields_no_chunks(self, small_chunks: None) -> None:
        fake = FakeTopicEmbeddings()
        chunks = SemanticStrategy(embeddings=fake).split(  # type: ignore[arg-type]
            "", source_meta=SOURCE_META, verbose=0
        )
        assert chunks == []
        assert fake.calls == []

    def test_invalid_verbose_raises(self, small_chunks: None) -> None:
        with pytest.raises(ValueError, match="verbose"):
            SemanticStrategy(embeddings=FakeTopicEmbeddings()).split(  # type: ignore[arg-type]
                "text", source_meta=SOURCE_META, verbose=9
            )


class TestDoclingHybridStrategy:
    def test_real_conversion_chunks_with_heading_paths(self, small_chunks: None) -> None:
        """One real (markdown-backend, no layout models) Docling conversion."""
        strategy = get_chunker("docling_hybrid")
        markdown = (
            "# Harbor Operations\n\n"
            "## Berthing\n\n"
            "The Gullwing ferry switches to battery power 800 meters before docking.\n\n"
            "## Dredging\n\n"
            "The dredger Moorhen cleared the channel to nine meters.\n"
        )
        chunks = strategy.split(markdown, source_meta=SOURCE_META, verbose=0)
        assert len(chunks) == 2
        assert [c.metadata["heading_path"] for c in chunks] == [
            "Harbor Operations > Berthing",
            "Harbor Operations > Dredging",
        ]
        assert "battery power" in chunks[0].page_content
        assert [c.metadata["chunk_index"] for c in chunks] == [0, 1]
        assert all(c.metadata["file_name"] == "a.md" for c in chunks)
        chunks[0].metadata["context"] = "chunk-0-only"
        assert "context" not in chunks[1].metadata
        # Deterministic: converting the same text again yields the same chunks.
        again = strategy.split(markdown, source_meta=SOURCE_META, verbose=0)
        assert [c.page_content for c in again] == [c.page_content for c in chunks]

    def test_invalid_verbose_raises(self, small_chunks: None) -> None:
        with pytest.raises(ValueError, match="verbose"):
            get_chunker("docling_hybrid").split("text", source_meta=SOURCE_META, verbose=7)


class TestWarnNearTokenCeiling:
    def test_warns_only_for_near_ceiling_chunks(self, caplog: pytest.LogCaptureFixture) -> None:
        from langchain_core.documents import Document

        big = Document(page_content=_words(520), metadata={"chunk_index": 3})
        small = Document(page_content="tiny", metadata={"chunk_index": 4})
        with caplog.at_level(logging.WARNING, logger="varagity.chunking.base"):
            warn_near_token_ceiling([small], strategy="token_based")
            assert not caplog.records
            warn_near_token_ceiling([big, small], strategy="token_based")
        assert len(caplog.records) == 1
        assert "'token_based'" in caplog.records[0].message
        assert "chunk 3" in caplog.records[0].message
