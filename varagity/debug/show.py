"""Rich-rendered console helpers backing the ``verbose=`` parameter convention.

Varagity separates three output channels (spec §14); this module implements
the first — *human-facing console output* — with three levels:

* ``0`` — off: render nothing.
* ``1`` — low: names and counts.
* ``2`` — high: full metadata, rich panels.

Conventions:

* Every public function in the codebase accepts
  ``verbose: int = settings.DEFAULT_VERBOSE`` and raises :class:`ValueError`
  on invalid levels (enforced via :func:`check_verbose`).
* All rendering lives here as ``v_<function_name>(...)`` helpers (e.g.
  ``v_discover``, ``v_chunk``, ``v_retrieve``), keeping presentation out of
  business logic. Helpers render nothing at level ``0``.

Concrete ``v_<name>`` helpers land alongside the features they render.
"""

from typing import TYPE_CHECKING

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.text import Text

if TYPE_CHECKING:  # imported for annotations only — avoids a runtime cycle
    from collections.abc import Sequence

    from langchain_core.documents import Document

    from varagity.ingest.discovery import Buckets
    from varagity.stores.records import RetrievedChunk

VERBOSE_LEVELS: tuple[int, ...] = (0, 1, 2)

console = Console()


def check_verbose(verbose: int) -> int:
    """Validate a ``verbose`` level.

    Called at the top of every function that accepts a ``verbose`` parameter,
    so an invalid level fails fast instead of silently rendering nothing.

    Args:
        verbose: Requested verbosity; must be 0 (off), 1 (low), or 2 (high).

    Returns:
        The validated level, unchanged.

    Raises:
        ValueError: If ``verbose`` is not one of :data:`VERBOSE_LEVELS`.
    """
    if verbose not in VERBOSE_LEVELS:
        raise ValueError(
            f"verbose must be one of {VERBOSE_LEVELS} (0=off, 1=low, 2=high); got {verbose!r}"
        )
    return verbose


def v_discover(buckets: "Buckets", verbose: int) -> None:
    """Render discovery results (for :func:`~varagity.ingest.discovery.discover_documents`).

    Args:
        buckets: The discovered corpus buckets.
        verbose: 0 = nothing; 1 = counts per bucket; 2 = also the file list.

    Raises:
        ValueError: If ``verbose`` is invalid.
    """
    check_verbose(verbose)
    if verbose == 0:
        return
    console.print(
        f"[bold]Discovered[/] {buckets.total} document(s) "
        f"({len(buckets.text_like)} text-like, {len(buckets.pdf)} pdf)"
    )
    if verbose == 2:
        for bucket_name, paths in (("text_like", buckets.text_like), ("pdf", buckets.pdf)):
            for path in paths:
                console.print(f"  [dim]{bucket_name}[/dim] {path}")


def v_retrieve(chunks: "Sequence[RetrievedChunk]", verbose: int) -> None:
    """Render retrieval results (for a retriever's ``retrieve``).

    Mirrors the reference's ``v_retrieve_docs``: at level 2 each retrieved
    chunk becomes a panel with its score, source, content, and (when the
    chunk was ingested with ``CONTEXTUALIZE`` on) its situating context.

    Args:
        chunks: The retrieved chunks, best first.
        verbose: 0 = nothing; 1 = retrieved count; 2 = a panel per chunk
            (score/source/content/context).

    Raises:
        ValueError: If ``verbose`` is invalid.
    """
    check_verbose(verbose)
    if verbose == 0:
        return
    console.print(f"[bold]Retrieved[/] {len(chunks)} chunk(s)")
    if verbose == 2:
        for rank, chunk in enumerate(chunks, start=1):
            source = str(chunk.metadata.get("source", "<unknown>"))
            page = chunk.metadata.get("page")
            body: RenderableType = Text(chunk.content)
            if chunk.context:
                body = Group(Panel(Text(chunk.context), title="context", style="dim"), body)
            console.print(
                Panel(
                    body,
                    title=f"match {rank} · score {chunk.score:.4f} · {chunk.chunk_id}",
                    subtitle=source if page is None else f"{source} · page {page}",
                    subtitle_align="left",
                )
            )


def v_situate_context(chunk_text: str, context: str, verbose: int) -> None:
    """Render one situating blurb (for :func:`~varagity.context.contextual.situate_context`).

    Level 1 stays quiet — the loader's contextualization sub-progress bar is
    the per-chunk signal there; level 2 shows each blurb next to a snippet of
    the chunk it situates.

    Args:
        chunk_text: The chunk being situated (snippet shown as the subtitle).
        context: The LLM-generated situating blurb.
        verbose: 0/1 = nothing; 2 = a panel per blurb.

    Raises:
        ValueError: If ``verbose`` is invalid.
    """
    check_verbose(verbose)
    if verbose < 2:
        return
    snippet = " ".join(chunk_text.split())
    if len(snippet) > 70:
        snippet = snippet[:69] + "…"
    console.print(
        Panel(
            Text(context) if context else Text("<empty blurb>", style="italic red"),
            title="context blurb",
            subtitle=f"chunk: {snippet}",
            subtitle_align="left",
            style="dim",
        )
    )


def v_chunk(chunks: "Sequence[Document]", verbose: int) -> None:
    """Render chunking results (for a chunking strategy's ``split``).

    Args:
        chunks: The chunks produced for one document (langchain
            ``Document`` objects with seeded metadata).
        verbose: 0 = nothing; 1 = file → chunk count; 2 = a panel per chunk
            with its full metadata.

    Raises:
        ValueError: If ``verbose`` is invalid.
    """
    check_verbose(verbose)
    if verbose == 0 or not chunks:
        return
    file_name = chunks[0].metadata.get("file_name", "<unknown>")
    console.print(f"[bold]{file_name}[/] → {len(chunks)} chunk(s)")
    if verbose == 2:
        for chunk in chunks:
            meta = ", ".join(f"{key}={value}" for key, value in sorted(chunk.metadata.items()))
            console.print(
                Panel(
                    Text(chunk.page_content),
                    title=f"chunk {chunk.metadata.get('chunk_index', '?')}",
                    subtitle=meta,
                    subtitle_align="left",
                )
            )
