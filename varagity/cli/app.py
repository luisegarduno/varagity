"""CLI argument parsing, subcommand dispatch, and the terminal Q&A loop.

Subcommands land with their vertical slices: ``ingest`` (Phase 3), ``chat``
(Phase 4 — the default when no subcommand is given), ``eval`` (Phase 9).
``chat`` follows the spec §13 startup sequence: ingest the corpus first,
then loop — prompt → retrieve → show matches → grounded answer — until
``:quit`` (or end-of-input, e.g. in a non-interactive container).
"""

import argparse
import logging

from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from varagity.config import get_settings
from varagity.debug.show import console
from varagity.generation.answer import answer_query
from varagity.ingest.loader import IngestSummary, ingest_corpus
from varagity.logging_setup import setup_logging
from varagity.models.registry import get_model
from varagity.retrieval import get_retriever
from varagity.stores.records import RetrievedChunk

logger = logging.getLogger(__name__)

# Typing this at the chat prompt exits the loop (spec §13).
QUIT_COMMAND = ":quit"

# Matches-table snippets are truncated to this many characters.
_SNIPPET_CHARS = 80


def _add_verbose_option(parser: argparse.ArgumentParser, default: object) -> None:
    """Attach the shared ``--verbose/-v`` option to a parser.

    Args:
        parser: The parser (top-level or subcommand) to extend.
        default: ``None`` on the top-level parser; ``argparse.SUPPRESS`` on
            subparsers so ``varagity -v 2 ingest`` isn't overwritten by the
            subparser's default when the flag follows the subcommand instead.
    """
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        choices=(0, 1, 2),
        default=default,
        help="console verbosity: 0=off, 1=low, 2=high (default: DEFAULT_VERBOSE from settings)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser.

    Returns:
        A parser accepting ``--verbose/-v`` (before or after the subcommand)
        and the registered subcommands; with no subcommand the app runs
        ``chat``.
    """
    parser = argparse.ArgumentParser(
        prog="varagity",
        description="Varagity — terminal RAG with Contextual Retrieval. "
        "Runs `chat` when no command is given.",
    )
    _add_verbose_option(parser, default=None)
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    ingest = subparsers.add_parser(
        "ingest",
        help="scan DOCS_PATH and ingest the corpus into both stores (pgvector + BM25)",
        description="Parse, chunk, contextualize, embed, and store every supported document "
        "under DOCS_PATH.",
    )
    _add_verbose_option(ingest, default=argparse.SUPPRESS)
    ingest.add_argument(
        "--reingest",
        action="store_true",
        help="delete and re-process every discovered document. Needed after pipeline-setting "
        "changes (CONTEXTUALIZE, chunk params): those don't change content hashes, so "
        "unchanged files are otherwise skipped",
    )
    chat = subparsers.add_parser(
        "chat",
        help="ingest the corpus, then answer questions from the terminal (the default)",
        description="Ingest DOCS_PATH, then loop: retrieve the top-k chunks per question "
        f"and generate a grounded, cited answer. Type {QUIT_COMMAND} to exit.",
    )
    _add_verbose_option(chat, default=argparse.SUPPRESS)
    return parser


def run(argv: list[str] | None = None) -> int:
    """Run the varagity command-line interface.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        The process exit code (``ingest`` returns 1 if any file failed).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)
    verbose = settings.DEFAULT_VERBOSE if args.verbose is None else args.verbose

    if args.command == "ingest":
        return _run_ingest(verbose, reingest=args.reingest)
    # chat is the default subcommand (spec §13).
    return _run_chat(verbose)


def _run_ingest(verbose: int, *, reingest: bool = False) -> int:
    """Execute the ``ingest`` subcommand.

    Args:
        verbose: Effective console verbosity.
        reingest: Delete and re-process every discovered document (the
            ``--reingest`` flag).

    Returns:
        ``0`` on success, ``1`` if any file failed to ingest.
    """
    summary = ingest_corpus(reingest=reingest, verbose=verbose)
    _show_ingest_summary(summary)
    return 1 if summary.failed else 0


def _run_chat(verbose: int) -> int:
    """Execute the ``chat`` subcommand: ingest on start, then the Q&A loop.

    A failed file during startup ingestion is logged and counted but does not
    block the loop — questions run against whatever the stores hold.

    Args:
        verbose: Effective console verbosity.

    Returns:
        ``0`` on clean exit (``:quit`` or end-of-input).
    """
    settings = get_settings()

    summary = ingest_corpus(verbose=verbose)
    _show_ingest_summary(summary)
    if summary.failed:
        logger.warning(
            "%d file(s) failed to ingest — answering from what was stored", summary.failed
        )

    # Resolve once so a misconfigured RETRIEVAL_METHOD fails before the loop.
    retriever = get_retriever(settings.RETRIEVAL_METHOD)
    llm = get_model("default")

    console.print(
        f"\nAsk a question ([bold]{QUIT_COMMAND}[/] to exit) — "
        f"retrieval: [bold]{settings.RETRIEVAL_METHOD}[/], top-{settings.TOP_K}\n"
    )
    while True:
        try:
            query = Prompt.ask("[bold cyan]varagity[/]", console=console)
        except (EOFError, KeyboardInterrupt):
            # Non-interactive stdin (e.g. the app container) or Ctrl-C/Ctrl-D.
            console.print()
            return 0
        query = query.strip()
        if not query:
            continue
        if query == QUIT_COMMAND:
            return 0
        state = answer_query(
            query,
            retriever=retriever,
            llm=llm,
            k=settings.TOP_K,
            verbose=verbose,
            on_retrieved=_show_matches,
        )
        console.print(Panel(Markdown(state["answer"]), title="answer", border_style="green"))


def _show_matches(chunks: list[RetrievedChunk]) -> None:
    """Render the retrieved matches as a table (spec §10.1 step 4).

    Args:
        chunks: The retrieved chunks, best first.
    """
    table = Table(title=f"Top {len(chunks)} matches")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Score", justify="right")
    table.add_column("Source", style="bold")
    table.add_column("Snippet")
    for rank, chunk in enumerate(chunks, start=1):
        source = str(chunk.metadata.get("file_name") or chunk.metadata.get("source", "<unknown>"))
        page = chunk.metadata.get("page")
        if page is not None:
            source = f"{source} p.{page}"
        table.add_row(str(rank), f"{chunk.score:.4f}", source, _snippet(chunk.content))
    console.print(table)


def _snippet(text: str, limit: int = _SNIPPET_CHARS) -> str:
    """Collapse whitespace and truncate text for one table cell.

    Args:
        text: The chunk content.
        limit: Maximum characters to keep.

    Returns:
        A single-line snippet, ellipsized when truncated.
    """
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"


def _show_ingest_summary(summary: IngestSummary) -> None:
    """Render the ingest run's counters as a table.

    Args:
        summary: The finished run's counters.
    """
    table = Table(title="Ingest summary")
    table.add_column("Metric", style="bold")
    table.add_column("Count", justify="right")
    table.add_row("files discovered", str(summary.discovered))
    table.add_row("files ingested", str(summary.ingested))
    table.add_row("chunks stored", str(summary.chunks))
    table.add_row("skipped (unchanged)", str(summary.skipped))
    table.add_row("no extractable text", str(summary.no_text))
    table.add_row("unsupported (no parser yet)", str(summary.unsupported))
    table.add_row("failed", str(summary.failed), style="red" if summary.failed else None)
    console.print(table)
