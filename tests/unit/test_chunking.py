"""Unit tests for the chunking registry and recursive_character strategy."""

from collections.abc import Callable

import pytest

from varagity.chunking import CHUNKER_REGISTRY, get_chunker
from varagity.chunking.recursive_character import RecursiveCharacterStrategy

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
    def test_recursive_character_registered(self) -> None:
        assert isinstance(get_chunker("recursive_character"), RecursiveCharacterStrategy)
        assert "recursive_character" in CHUNKER_REGISTRY

    def test_unknown_name_raises_listing_available(self) -> None:
        with pytest.raises(KeyError, match="recursive_character"):
            get_chunker("semantic")


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
