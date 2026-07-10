"""CLI argument parsing and subcommand dispatch.

Subcommands land with their vertical slices: ``ingest`` (Phase 3), ``chat``
(Phase 4, becomes the default), ``eval`` (Phase 9). With no subcommand the
app prints help plus the loaded settings, so ``uv run main.py`` always works.
"""

import argparse

from rich.table import Table

from varagity.config import Settings, get_settings
from varagity.debug.show import console
from varagity.ingest.loader import IngestSummary, ingest_corpus
from varagity.logging_setup import setup_logging


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
        and the registered subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="varagity",
        description="Varagity — terminal RAG with Contextual Retrieval.",
    )
    _add_verbose_option(parser, default=None)
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    ingest = subparsers.add_parser(
        "ingest",
        help="scan DOCS_PATH and ingest the corpus into the vector store",
        description="Parse, chunk, embed, and store every supported document under DOCS_PATH.",
    )
    _add_verbose_option(ingest, default=argparse.SUPPRESS)
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
        return _run_ingest(verbose)

    # No subcommand yet (`chat` becomes the default in Phase 4): show help
    # and the loaded configuration.
    parser.print_help()
    console.print()
    _show_settings(settings, verbose)
    return 0


def _run_ingest(verbose: int) -> int:
    """Execute the ``ingest`` subcommand.

    Args:
        verbose: Effective console verbosity.

    Returns:
        ``0`` on success, ``1`` if any file failed to ingest.
    """
    summary = ingest_corpus(verbose=verbose)
    _show_ingest_summary(summary)
    return 1 if summary.failed else 0


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


# Settings whose names match these markers are redacted in console output.
_SECRET_MARKERS = ("PASSWORD", "KEY", "SECRET", "TOKEN")


def _show_settings(settings: Settings, verbose: int) -> None:
    """Render the loaded settings as a table, redacting secret-like values.

    Args:
        settings: The loaded application settings.
        verbose: Effective console verbosity for this invocation.
    """
    table = Table(title="Loaded settings")
    table.add_column("Setting", style="bold")
    table.add_column("Value")
    for name, value in settings.model_dump().items():
        redact = any(marker in name.upper() for marker in _SECRET_MARKERS)
        table.add_row(name, "'***'" if redact else repr(value))
    table.add_row("(effective verbose)", repr(verbose))
    console.print(table)
