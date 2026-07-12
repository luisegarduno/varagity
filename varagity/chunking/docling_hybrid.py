"""Docling ``HybridChunker`` as a registered strategy (spec_v2 §7).

The alternative ``markdown_aware`` implementation the v2 plan benchmarks:
Docling's tokenization- *and* structure-aware chunker, shipped as its own
registry entry so the Phase 6 eval sweep compares the two head-to-head and
the default is benchmark-decided (ADR-008), not hand-picked.

The strategy re-parses the parser's markdown into a ``DoclingDocument``
(in-memory ``DocumentStream``, markdown backend — declarative, no layout
models) and runs ``HybridChunker`` over the document object model: one chunk
per document item, oversized items token-split, undersized peers under the
same headings merged (``merge_peers`` — the packing trade ``markdown_aware``
declines). Heading provenance lands in ``heading_path`` exactly like
``markdown_aware``. Sizing: ``CHUNK_SIZE`` is the tokenizer's ``max_tokens``
budget (**tokens**, tiktoken ``cl100k_base`` — the codebase-wide documented
approximation); ``CHUNK_OVERLAP`` has no ``HybridChunker`` equivalent and is
ignored.

Chunk text is the item text (``chunk.text``), **not** the chunker's
heading-prefixed ``contextualize()`` rendering: Contextual Retrieval already
prepends an LLM situating blurb at ingest (spec §9.4), and stacking two
context prefixes would double-count headings and eat the 512-token budget.

Docling imports are deferred to call time (the parser-module convention):
importing this module on CLI start must not pay for Docling's machinery.
"""

from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document

from varagity.chunking.base import register, warn_near_token_ceiling
from varagity.config import get_settings
from varagity.debug.show import check_verbose, v_chunk

if TYPE_CHECKING:  # heavy imports, type-only (runtime imports are lazy)
    from docling.chunking import HybridChunker
    from docling.document_converter import DocumentConverter

# Separator rendered between heading levels (matches markdown_aware).
_PATH_SEPARATOR = " > "

# Synthetic stream name for the in-memory markdown re-parse (the .md suffix
# selects Docling's markdown backend).
_STREAM_NAME = "chunking-input.md"


@register("docling_hybrid")
class DoclingHybridStrategy:
    """Docling ``HybridChunker`` over the re-parsed document structure."""

    def __init__(self) -> None:
        """Initialize the converter/chunker caches (nothing heavy here)."""
        self._converter: DocumentConverter | None = None
        self._chunker: HybridChunker | None = None
        self._chunker_max_tokens: int | None = None

    def split(
        self, text: str, *, source_meta: dict[str, Any], verbose: int | None = None
    ) -> list[Document]:
        """Split a document's markdown with Docling's ``HybridChunker``.

        Args:
            text: The full document text (markdown from the rich-format
                parsers; plain text parses as markdown paragraphs).
            source_meta: Provenance copied into every chunk's metadata.
            verbose: Console verbosity (0–2); defaults to
                ``settings.DEFAULT_VERBOSE``.

        Returns:
            The chunks in document order, each carrying ``source_meta``,
            its ``chunk_index``, and — when Docling attributes headings —
            the ``heading_path`` breadcrumb.

        Raises:
            ValueError: If ``verbose`` is invalid.
        """
        from io import BytesIO

        from docling_core.types.io import DocumentStream

        settings = get_settings()
        verbose = check_verbose(settings.DEFAULT_VERBOSE if verbose is None else verbose)
        converted = self._convert(
            DocumentStream(name=_STREAM_NAME, stream=BytesIO(text.encode("utf-8")))
        )
        chunker = self._hybrid_chunker(settings.CHUNK_SIZE)  # CHUNK_SIZE as tokens
        chunks: list[Document] = []
        for chunk_index, item in enumerate(chunker.chunk(converted)):
            metadata: dict[str, Any] = dict(source_meta)
            headings = getattr(item.meta, "headings", None)
            if headings:
                metadata["heading_path"] = _PATH_SEPARATOR.join(headings)
            metadata["chunk_index"] = chunk_index
            chunks.append(Document(page_content=item.text, metadata=metadata))
        warn_near_token_ceiling(chunks, strategy="docling_hybrid")
        v_chunk(chunks, verbose=verbose)
        return chunks

    def _convert(self, stream: Any) -> Any:
        """Re-parse markdown text into a ``DoclingDocument`` (cached converter).

        Args:
            stream: The in-memory ``DocumentStream`` to convert.

        Returns:
            The converted Docling document.
        """
        if self._converter is None:
            from docling.datamodel.base_models import InputFormat
            from docling.document_converter import DocumentConverter

            # Markdown-only: the input is always the parsers' markdown/plain
            # text, and restricting formats skips every heavyweight pipeline.
            self._converter = DocumentConverter(allowed_formats=[InputFormat.MD])
        return self._converter.convert(stream).document

    def _hybrid_chunker(self, max_tokens: int) -> "HybridChunker":
        """Build (or reuse) the ``HybridChunker`` for the configured budget.

        Args:
            max_tokens: Token budget per chunk (``CHUNK_SIZE``, as tokens).

        Returns:
            The configured chunker (rebuilt only when the budget changes,
            e.g. under the eval harness's pinned settings).
        """
        if self._chunker is None or self._chunker_max_tokens != max_tokens:
            import tiktoken
            from docling.chunking import HybridChunker
            from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

            tokenizer = OpenAITokenizer(
                tokenizer=tiktoken.get_encoding("cl100k_base"), max_tokens=max_tokens
            )
            self._chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
            self._chunker_max_tokens = max_tokens
        return self._chunker
