"""Heading-aware markdown splitting (spec_v2 §7).

Every rich-format parser (PDF, office, web) exports **structure-aware
markdown** through the shared Docling core — this strategy finally exploits
that structure: split at ATX headings (``#`` … ``######``), keep each
section's text together, and carry the section's heading breadcrumb into
chunk metadata as ``heading_path`` (e.g. ``"Harbor Operations > Dredging >
Channel depth"``), where the JSONB metadata column absorbs it with no schema
migration and the evidence panel can surface it as breadcrumb context.

This is the from-scratch header splitter the v2 plan benchmarks against
Docling's ``HybridChunker`` (shipped alongside as ``docling_hybrid``; the
Phase 6 sweep decides the default — ADR-008). Behavior notes:

* Headings inside fenced code blocks (three-plus backticks or tildes) are
  content, not structure.
* A section's own heading line stays in its chunk text (retrieval-visible);
  ancestors live only in ``heading_path``.
* Sections are **not** merged across headings — coherence over packing;
  ``HybridChunker`` makes the opposite trade, which is exactly what the
  sweep measures.
* A section longer than ``CHUNK_SIZE`` (**characters**, the v1 unit — this
  strategy sizes like ``recursive_character``) is re-split recursively;
  every sub-chunk inherits the section's ``heading_path``.
* Text before the first heading forms a section with no ``heading_path``.
"""

import re
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from varagity.chunking.base import register, warn_near_token_ceiling
from varagity.config import get_settings
from varagity.debug.show import check_verbose, v_chunk

# ATX heading: 1-6 hashes, a space, then the title (CommonMark).
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")

# Fence open/close: three-plus backticks or tildes (CommonMark fenced code).
_FENCE_RE = re.compile(r"^(`{3,}|~{3,})")

# Separator rendered between heading levels in the metadata breadcrumb.
_PATH_SEPARATOR = " > "


def _sections(text: str) -> list[tuple[tuple[str, ...], str]]:
    """Split markdown into heading-scoped sections.

    Args:
        text: The document's markdown text.

    Returns:
        ``(heading_path, section_text)`` pairs in document order. The path
        includes the section's own heading; preamble text before the first
        heading gets an empty path. Section text keeps its heading line and
        original line breaks; blank-only sections (a heading immediately
        followed by another heading contributes no section of its own — its
        title survives in descendants' paths) are dropped.
    """
    sections: list[tuple[tuple[str, ...], str]] = []
    path: tuple[str, ...] = ()
    lines: list[str] = []
    has_content = False
    in_fence = False
    fence_marker = ""

    def flush() -> None:
        nonlocal lines, has_content
        if has_content:
            sections.append((path, "\n".join(lines).strip("\n")))
        lines = []
        has_content = False

    for line in text.split("\n"):
        fence = _FENCE_RE.match(line)
        if fence:
            marker = fence.group(1)[0] * 3
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif marker == fence_marker:
                in_fence = False
        heading = None if in_fence else _HEADING_RE.match(line)
        if heading is None:
            lines.append(line)
            has_content = has_content or bool(line.strip())
            continue
        flush()
        level, title = len(heading.group(1)), heading.group(2)
        path = (*path[: level - 1], title)
        lines = [line]  # the heading line belongs to its own section's text
    flush()
    return sections


@register("markdown_aware")
class MarkdownAwareStrategy:
    """Splits on ATX headings, carrying the heading breadcrumb as metadata."""

    def split(
        self, text: str, *, source_meta: dict[str, Any], verbose: int | None = None
    ) -> list[Document]:
        """Split a document's markdown into heading-coherent chunks.

        Args:
            text: The full document text (markdown from the rich-format
                parsers; plain text degrades to one section, then the
                recursive re-split).
            source_meta: Provenance copied into every chunk's metadata.
            verbose: Console verbosity (0–2); defaults to
                ``settings.DEFAULT_VERBOSE``.

        Returns:
            The chunks in document order, each carrying ``source_meta``,
            its ``chunk_index``, and — for sections under a heading — the
            ``heading_path`` breadcrumb.

        Raises:
            ValueError: If ``verbose`` is invalid.
        """
        settings = get_settings()
        verbose = check_verbose(settings.DEFAULT_VERBOSE if verbose is None else verbose)
        resplitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,  # characters, like recursive_character
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        chunks: list[Document] = []
        for path, section_text in _sections(text):
            texts = (
                resplitter.split_text(section_text)
                if len(section_text) > settings.CHUNK_SIZE
                else [section_text]
            )
            for piece in texts:
                metadata: dict[str, Any] = dict(source_meta)
                if path:
                    metadata["heading_path"] = _PATH_SEPARATOR.join(path)
                chunks.append(Document(page_content=piece, metadata=metadata))
        for chunk_index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = chunk_index
        warn_near_token_ceiling(chunks, strategy="markdown_aware")
        v_chunk(chunks, verbose=verbose)
        return chunks
