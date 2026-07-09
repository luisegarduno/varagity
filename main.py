"""Varagity entrypoint — a thin ``argparse`` shell.

Subcommands (``ingest``, ``chat``, ``eval``) register here as their vertical
slices land; parsing then delegates to ``varagity.cli``. Until the first
subcommand exists, invoking prints help plus the loaded settings so
``uv run main.py`` always works.
"""

import argparse

from rich.console import Console
from rich.table import Table

from varagity.config import Settings, get_settings
from varagity.logging_setup import setup_logging


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser.

    Returns:
        A parser accepting ``--verbose/-v`` and a subcommand slot (no
        subcommands registered yet).
    """
    parser = argparse.ArgumentParser(
        prog="varagity",
        description="Varagity — terminal RAG with Contextual Retrieval.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        choices=(0, 1, 2),
        default=None,
        help="console verbosity: 0=off, 1=low, 2=high (default: DEFAULT_VERBOSE from settings)",
    )
    parser.add_subparsers(dest="command", metavar="COMMAND")
    return parser


def _show_settings(settings: Settings, verbose: int, console: Console) -> None:
    """Render the loaded settings as a table.

    Args:
        settings: The loaded application settings.
        verbose: Effective console verbosity for this invocation.
        console: Rich console to render to.
    """
    table = Table(title="Loaded settings")
    table.add_column("Setting", style="bold")
    table.add_column("Value")
    for name, value in settings.model_dump().items():
        table.add_row(name, repr(value))
    table.add_row("(effective verbose)", repr(verbose))
    console.print(table)


def main(argv: list[str] | None = None) -> int:
    """Run the varagity command-line interface.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        The process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)
    verbose = settings.DEFAULT_VERBOSE if args.verbose is None else args.verbose

    # No subcommands are registered yet (`ingest` arrives with the ingestion
    # slice): show help and the loaded configuration.
    console = Console()
    parser.print_help()
    console.print()
    _show_settings(settings, verbose, console)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
