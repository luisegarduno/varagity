"""CLI argument parsing, subcommand dispatch, and the terminal Q&A loop.

Subcommands land with their vertical slices: ``ingest`` (Phase 3), ``chat``
(Phase 4 — the default when no subcommand is given), ``eval`` / ``eval
ocr`` (Phase 9). ``chat`` follows the spec §13 startup sequence: ingest the
corpus first, then loop — prompt → retrieve → show matches → grounded
answer — until ``:quit`` (or end-of-input, e.g. in a non-interactive
container).

Since Phase 8 every subcommand runs through the Prefect flows
(``varagity.pipeline``), invoked directly in-process — no worker or
deployment (spec §21 #8) — so every ingest stage and every question is a
tracked run at the Prefect UI (``:4200``).
"""

import argparse
import logging
from typing import Any

from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from varagity.config import get_settings
from varagity.debug.show import console, trace_badges
from varagity.ingest.loader import IngestSummary
from varagity.logging_setup import setup_logging
from varagity.models.registry import get_model
from varagity.pipeline import eval_flow, ingest_flow, ocr_benchmark_flow, query_flow
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
    evaluate = subparsers.add_parser(
        "eval",
        help="measure retrieval quality (recall@k/pass@k, 5-config matrix) on ephemeral stores",
        description="Run the spec §16 evaluation harness against ephemeral testcontainers "
        "stores (Docker required) and the live GPU services. Without a target, runs the "
        "5-configuration retrieval matrix; `eval ocr` benchmarks the OCR engines.",
    )
    _add_verbose_option(evaluate, default=argparse.SUPPRESS)
    eval_targets = evaluate.add_subparsers(dest="eval_command", metavar="TARGET")
    eval_ocr = eval_targets.add_parser(
        "ocr",
        help="benchmark the OCR engines (CER/WER, pages/sec, retrieval recall)",
        description="Parse the scanned fixture PDFs with every OCR engine, score them "
        "against ground truth, and measure the retrieval impact per engine.",
    )
    _add_verbose_option(eval_ocr, default=argparse.SUPPRESS)
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
    if args.command == "eval":
        if getattr(args, "eval_command", None) == "ocr":
            return _run_eval_ocr(verbose)
        return _run_eval(verbose)
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
    summary = ingest_flow(reingest=reingest, verbose=verbose)
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

    summary = ingest_flow(verbose=verbose)
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
        state = query_flow(
            query,
            retriever=retriever,
            llm=llm,
            k=settings.TOP_K,
            verbose=verbose,
            on_retrieved=_show_matches,
        )
        console.print(Panel(Markdown(state["answer"]), title="answer", border_style="green"))


def _run_eval(verbose: int) -> int:
    """Execute the ``eval`` subcommand: the 5-configuration retrieval matrix.

    Args:
        verbose: Effective console verbosity.

    Returns:
        ``0`` on success (a failed run raises).
    """
    results = eval_flow(verbose=verbose)
    _show_matrix_results(results)
    return 0


def _run_eval_ocr(verbose: int) -> int:
    """Execute the ``eval ocr`` subcommand: the OCR engine benchmark.

    Args:
        verbose: Effective console verbosity.

    Returns:
        ``0`` on success (a failed run raises).
    """
    results = ocr_benchmark_flow(verbose=verbose)
    _show_ocr_results(results)
    return 0


# Matrix config keys in ladder order, with their table labels (spec §16 +
# spec_v2 §5.5).
_MATRIX_CONFIG_LABELS: tuple[tuple[str, str], ...] = (
    ("semantic_noncontextual", "1. semantic, non-contextual"),
    ("semantic_contextual", "2. semantic, contextual"),
    ("bm25_contextual", "3. BM25, contextual"),
    ("hybrid_contextual", "4. hybrid, contextual"),
    ("hybrid_rerank_contextual", "5. hybrid + rerank, contextual"),
)


def _show_matrix_results(results: dict[str, Any]) -> None:
    """Render the retrieval matrix and the chunker sweep as tables.

    Args:
        results: The :func:`varagity.eval.evaluate.run_matrix` document.
    """
    k_values: list[int] = results["k_values"]
    table = Table(
        title=f"Retrieval matrix — {results['n_queries']} golden queries, "
        f"{results['chunks_ingested']} chunks"
    )
    table.add_column("Configuration", style="bold")
    for k in k_values:
        table.add_column(f"recall@{k}", justify="right")
    for k in k_values:
        table.add_column(f"pass@{k}", justify="right", style="dim")
    for key, label in _MATRIX_CONFIG_LABELS:
        scores = results["configs"][key]
        table.add_row(
            label,
            *(f"{scores['recall'][str(k)]:.3f}" for k in k_values),
            *(f"{scores['pass'][str(k)]:.3f}" for k in k_values),
        )
    console.print(table)
    _show_chunker_sweep(results)
    console.print(f"Results written to [bold]{results['results_path']}[/]")


def _show_chunker_sweep(results: dict[str, Any]) -> None:
    """Render the chunker sweep as a strategy × method table (spec_v2 §7.4).

    Args:
        results: The :func:`varagity.eval.evaluate.run_matrix` document
            (older result files without a sweep render nothing).
    """
    sweep: dict[str, Any] = results.get("chunker_sweep") or {}
    if not sweep:
        return
    k_values: list[int] = results["k_values"]
    table = Table(title="Chunker sweep — contextual ingest per strategy, fact-anchored golden refs")
    table.add_column("Strategy", style="bold")
    table.add_column("Chunks", justify="right")
    table.add_column("Ingest s", justify="right")
    table.add_column("Method")
    for k in k_values:
        table.add_column(f"recall@{k}", justify="right")
    for k in k_values:
        table.add_column(f"pass@{k}", justify="right", style="dim")
    for strategy, data in sweep.items():
        for row, (method, scores) in enumerate(data["configs"].items()):
            table.add_row(
                strategy if row == 0 else "",
                str(data["chunks"]) if row == 0 else "",
                f"{data['ingest_seconds']:.1f}" if row == 0 else "",
                method,
                *(f"{scores['recall'][str(k)]:.3f}" for k in k_values),
                *(f"{scores['pass'][str(k)]:.3f}" for k in k_values),
            )
    console.print(table)
    for strategy, data in sweep.items():
        if data["unresolved_facts"]:
            console.print(
                f"[yellow]{strategy}: {len(data['unresolved_facts'])} golden fact(s) not "
                f"found in any chunk (guaranteed misses): {data['unresolved_facts']}[/]"
            )


def _show_ocr_results(results: dict[str, Any]) -> None:
    """Render the OCR benchmark: intrinsic quality and retrieval impact.

    Args:
        results: The :func:`varagity.eval.ocr_benchmark.run_ocr_benchmark`
            document.
    """
    k_values: list[int] = results["k_values"]
    intrinsic = Table(
        title=f"OCR intrinsic quality — {len(results['fixtures'])} fixture PDFs "
        "(normalized CER/WER vs ground truth)"
    )
    intrinsic.add_column("Engine", style="bold")
    intrinsic.add_column("CER", justify="right")
    intrinsic.add_column("WER", justify="right")
    intrinsic.add_column("Pages/s", justify="right")
    for engine, data in results["engines"].items():
        overall = data["intrinsic"]["overall"]
        intrinsic.add_row(
            engine,
            f"{overall['cer']:.4f}",
            f"{overall['wer']:.4f}",
            f"{overall['pages_per_sec']:.3f}",
        )
    console.print(intrinsic)

    retrieval = Table(
        title=f"OCR retrieval impact — {results['n_scanned_queries']} scanned-doc queries "
        "(non-contextual ingest per engine)"
    )
    retrieval.add_column("Engine", style="bold")
    retrieval.add_column("Method")
    for k in k_values:
        retrieval.add_column(f"recall@{k}", justify="right")
    for engine, data in results["engines"].items():
        unresolvable = data["retrieval"]["unresolvable_golden_refs"]
        for method, scores in data["retrieval"]["methods"].items():
            retrieval.add_row(
                engine,
                method,
                *(f"{scores['recall'][str(k)]:.3f}" for k in k_values),
            )
        if unresolvable:
            console.print(
                f"[yellow]{engine}: {len(unresolvable)} golden ref(s) unresolvable under this "
                f"engine's chunk boundaries (counted as misses): {unresolvable}[/]"
            )
    console.print(retrieval)
    console.print(f"Results written to [bold]{results['results_path']}[/]")


def _show_matches(chunks: list[RetrievedChunk]) -> None:
    """Render the retrieved matches as a table (spec §10.1 step 4).

    Args:
        chunks: The retrieved chunks, best first (the ``Trace`` column shows
            each chunk's compact rank provenance when the retriever attached
            one — spec_v2 §9.2).
    """
    table = Table(title=f"Top {len(chunks)} matches")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Score", justify="right")
    table.add_column("Source", style="bold")
    table.add_column("Trace", style="dim")
    table.add_column("Snippet")
    for rank, chunk in enumerate(chunks, start=1):
        source = str(chunk.metadata.get("file_name") or chunk.metadata.get("source", "<unknown>"))
        page = chunk.metadata.get("page")
        if page is not None:
            source = f"{source} p.{page}"
        trace = trace_badges(chunk.trace) if chunk.trace is not None else "—"
        table.add_row(str(rank), f"{chunk.score:.4f}", source, trace, _snippet(chunk.content))
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
