"""Token-denominated recursive splitting (spec_v2 §7, v2 plan decision #12).

Wraps langchain's ``RecursiveCharacterTextSplitter.from_tiktoken_encoder``:
the same separator hierarchy as ``recursive_character`` (paragraph → line →
word → char), but ``CHUNK_SIZE`` / ``CHUNK_OVERLAP`` **count tokens, not
characters** — aligning chunk budgets to the e5 embedder's 512-token ceiling
instead of the character approximation the v1 strategy documents.

Token counting uses tiktoken's ``cl100k_base`` — the codebase-wide
*documented approximation* of e5's actual XLM-RoBERTa vocabulary (see
``varagity.tokens``). An exact counter (the e5 HF tokenizer as a custom
``length_function``) plugs into the ``__init__`` seam without touching the
registry; it was evaluated against the approximation in the chunker sweep
rather than shipped (an extra ``transformers`` hot path + first-run
tokenizer download for a small counting delta — ADR-008).

Note the langchain merge nuance: the splitter sizes chunks by summing piece
lengths, and re-tokenizing joined text can differ by a token or two at the
seams, so real-tokenizer chunks may exceed the budget by a whisker. The
≥480 warning (not a hard 512 assert) is the guard that matters.
"""

from collections.abc import Callable
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from varagity.chunking.base import register, warn_near_token_ceiling
from varagity.config import get_settings
from varagity.debug.show import check_verbose, v_chunk

# The tiktoken encoding used for sizing (matches varagity.tokens).
_ENCODING_NAME = "cl100k_base"


@register("token_based")
class TokenBasedStrategy:
    """Recursive separator splitting with token-denominated sizes."""

    def __init__(self, *, length_function: Callable[[str], int] | None = None) -> None:
        """Create the strategy.

        Args:
            length_function: Alternative token counter (e.g. an exact e5 HF
                tokenizer ``lambda``, or a fake in tests). Defaults to the
                tiktoken ``cl100k_base`` counter via
                ``from_tiktoken_encoder``.
        """
        self._length_function = length_function

    def _splitter(self, chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
        """Build the token-sized splitter for the configured budgets.

        Args:
            chunk_size: Chunk budget, in tokens.
            chunk_overlap: Overlap between consecutive chunks, in tokens.

        Returns:
            The configured splitter.
        """
        if self._length_function is not None:
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=self._length_function,
            )
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=_ENCODING_NAME,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(
        self, text: str, *, source_meta: dict[str, Any], verbose: int | None = None
    ) -> list[Document]:
        """Split a document's text into overlapping token-budgeted chunks.

        Args:
            text: The full document text.
            source_meta: Provenance copied into every chunk's metadata (the
                splitter deep-copies per chunk).
            verbose: Console verbosity (0–2); defaults to
                ``settings.DEFAULT_VERBOSE``.

        Returns:
            The chunks in document order, each carrying ``source_meta`` plus
            its ``chunk_index``.

        Raises:
            ValueError: If ``verbose`` is invalid.
        """
        settings = get_settings()
        verbose = check_verbose(settings.DEFAULT_VERBOSE if verbose is None else verbose)
        splitter = self._splitter(settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)  # tokens, not chars
        chunks = splitter.create_documents([text], metadatas=[dict(source_meta)])
        for chunk_index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = chunk_index
        warn_near_token_ceiling(chunks, strategy="token_based")
        v_chunk(chunks, verbose=verbose)
        return chunks
