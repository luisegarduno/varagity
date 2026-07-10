"""Default v1 chunking strategy: recursive character splitting (spec §9.3).

Wraps langchain's ``RecursiveCharacterTextSplitter`` with the configured
``CHUNK_SIZE`` / ``CHUNK_OVERLAP``. **Both parameters count characters, not
tokens** (the splitter's ``length_function`` defaults to ``len``); a
token-based strategy would be a new registry file, not an edit here.
"""

from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from varagity.chunking.base import register
from varagity.config import get_settings
from varagity.debug.show import check_verbose, v_chunk


@register("recursive_character")
class RecursiveCharacterStrategy:
    """Splits on a separator hierarchy (paragraph → line → word → char)."""

    def split(
        self, text: str, *, source_meta: dict[str, Any], verbose: int | None = None
    ) -> list[Document]:
        """Split a document's text into overlapping character chunks.

        Args:
            text: The full document text.
            source_meta: Provenance copied into every chunk's metadata (the
                splitter deep-copies per chunk, so later per-chunk fields
                don't bleed across siblings).
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
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,  # characters, not tokens (spec §9.3)
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        chunks = splitter.create_documents([text], metadatas=[dict(source_meta)])
        for chunk_index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = chunk_index
        v_chunk(chunks, verbose=verbose)
        return chunks
