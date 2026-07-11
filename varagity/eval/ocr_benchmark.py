"""OCR engine benchmark: EasyOCR vs Tesseract (plan decision #10, spec §16).

Phase 7 shipped EasyOCR as a *provisional* fallback engine behind the
pluggable ``OCR_ENGINE`` factory; this benchmark supplies the data that
picks the shipped default (recorded in ADR-004). Two measurements per
engine:

- **Intrinsic quality** — the scanned/mixed fixture PDFs are parsed through
  the real two-pass parser and compared against their known ground-truth
  text (``data/eval/ocr_truth/``, transcribed from the images the fixtures
  were generated with): **CER/WER** via ``jiwer`` after normalization
  (lowercase, punctuation stripped — which also drops Docling's markdown
  markup — whitespace collapsed), plus **pages/sec** wall-clock over a
  warmed engine (one untimed warm-up conversion loads models first).

- **Retrieval impact** (supplementary) — the full fixtures corpus is
  ingested once per engine into ephemeral testcontainers stores (real
  embeddings; ``CONTEXTUALIZE`` off so the engine is the only variable),
  then recall@k / pass@k are reported for the scanned-document golden
  queries. The non-scanned corpus documents ride along as distractors —
  without them every query would trivially retrieve the whole index.
  OCR noise hits BM25 keyword matching hardest, so all three retrieval
  methods are reported. Directional on a tiny corpus, per the plan; a
  golden ref an engine's chunk boundaries can't resolve is counted as a
  guaranteed miss and reported.

``jiwer`` lives in the ``eval`` dependency group; its import is deferred
to call time like the group's other members.
"""

import logging
import string
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from varagity.config import get_settings
from varagity.debug.show import check_verbose
from varagity.eval.containers import ephemeral_stores
from varagity.eval.datasets import ResolvedGoldenEntry, load_golden, resolve_golden
from varagity.eval.evaluate import (
    EVAL_CORPUS,
    GOLDEN_PATH,
    K_VALUES,
    RESULTS_DIR,
    IngestCallable,
    measure_retriever,
    pinned_eval_settings,
    validate_golden_against_store,
    write_results,
)
from varagity.ingest.loader import ingest_corpus
from varagity.ingest.parsers import get_parser
from varagity.models.registry import get_model
from varagity.retrieval.base import Retriever
from varagity.retrieval.bm25 import BM25Retriever
from varagity.retrieval.hybrid import HybridRetriever
from varagity.retrieval.semantic import SemanticRetriever

logger = logging.getLogger(__name__)

# The engines under comparison — the two the OCR_ENGINE factory ships
# (varagity.ingest.parsers.pdf.OCR_ENGINE_FACTORIES).
ENGINES: tuple[str, ...] = ("easyocr", "tesseract")

# Ground-truth transcriptions of the OCR fixtures (tracked in git).
OCR_TRUTH_DIR = Path("data/eval/ocr_truth")

# The corpus fixtures that exercise OCR (scanned + mixed), with their page
# counts — fixture constants, like the planted facts the tests assert on.
OCR_FIXTURE_PAGES: dict[str, int] = {
    "moorhen_dredging_memo.pdf": 1,
    "breakwater_survey.pdf": 2,
}


def normalize_ocr_text(text: str) -> str:
    """Normalize text for OCR error metrics.

    Lowercases, strips punctuation (which also removes Docling's markdown
    markup — ``#`` headings, table pipes — so the comparison measures
    recovered *text*, not export formatting), and collapses whitespace.
    Applied identically to ground truth and hypothesis.

    Args:
        text: Raw ground-truth or extracted text.

    Returns:
        The normalized text.
    """
    cleaned = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(cleaned.split())


def character_error_rate(truth: str, hypothesis: str) -> float:
    """CER between normalized ground truth and OCR output.

    Args:
        truth: Ground-truth text (non-empty after normalization).
        hypothesis: The engine's extracted text.

    Returns:
        The character error rate (0.0 = perfect; can exceed 1.0).

    Raises:
        ValueError: If the normalized ground truth is empty.
    """
    import jiwer

    reference = normalize_ocr_text(truth)
    if not reference:
        raise ValueError("ground truth is empty after normalization")
    return float(jiwer.cer(reference, normalize_ocr_text(hypothesis)))


def word_error_rate(truth: str, hypothesis: str) -> float:
    """WER between normalized ground truth and OCR output.

    Args:
        truth: Ground-truth text (non-empty after normalization).
        hypothesis: The engine's extracted text.

    Returns:
        The word error rate (0.0 = perfect; can exceed 1.0).

    Raises:
        ValueError: If the normalized ground truth is empty.
    """
    import jiwer

    reference = normalize_ocr_text(truth)
    if not reference:
        raise ValueError("ground truth is empty after normalization")
    return float(jiwer.wer(reference, normalize_ocr_text(hypothesis)))


def _scanned_entries(entries: Sequence[ResolvedGoldenEntry]) -> list[ResolvedGoldenEntry]:
    """Filter the golden entries whose relevant chunks are all OCR-fixture ones.

    Args:
        entries: The full resolved golden set.

    Returns:
        The scanned-document queries (the retrieval-impact denominator).
    """
    return [
        entry
        for entry in entries
        if all(ref.rel_source in OCR_FIXTURE_PAGES for ref in entry.relevant)
    ]


def _measure_intrinsic(engine: str, corpus_root: Path, verbose: int) -> dict[str, Any]:
    """Parse the OCR fixtures with one engine and score against ground truth.

    One untimed warm-up conversion loads the engine's models (and Docling's
    layout models) so the timed passes measure conversion, not first-run
    downloads/initialization.

    Args:
        engine: The ``OCR_ENGINE`` registry name.
        corpus_root: Directory holding the OCR fixtures.
        verbose: Validated console verbosity.

    Returns:
        Per-fixture and overall CER/WER plus pages/sec.
    """
    parser = get_parser("pdf")
    fixtures = sorted(OCR_FIXTURE_PAGES)
    warmup = corpus_root / fixtures[0]
    logger.info("[%s] warm-up conversion of %s (untimed)", engine, warmup.name)
    parser.extract(warmup, verbose=0)

    per_fixture: dict[str, Any] = {}
    truths: list[str] = []
    extractions: list[str] = []
    total_seconds = 0.0
    total_pages = 0
    for name in fixtures:
        truth = (OCR_TRUTH_DIR / name).with_suffix(".txt").read_text(encoding="utf-8")
        started = time.monotonic()
        raw = parser.extract(corpus_root / name, verbose=0)
        seconds = time.monotonic() - started
        pages = OCR_FIXTURE_PAGES[name]
        total_seconds += seconds
        total_pages += pages
        truths.append(normalize_ocr_text(truth))
        extractions.append(normalize_ocr_text(raw.text))
        per_fixture[name] = {
            "cer": round(character_error_rate(truth, raw.text), 4),
            "wer": round(word_error_rate(truth, raw.text), 4),
            "seconds": round(seconds, 2),
            "pages": pages,
            "extraction": raw.source_meta.get("extraction"),
        }
        logger.info(
            "[%s] %s: CER %.4f, WER %.4f, %.2fs",
            engine,
            name,
            per_fixture[name]["cer"],
            per_fixture[name]["wer"],
            seconds,
        )

    import jiwer

    return {
        "per_fixture": per_fixture,
        "overall": {
            # Corpus-level rates weight every fixture by its true length.
            "cer": round(float(jiwer.cer(truths, extractions)), 4),
            "wer": round(float(jiwer.wer(truths, extractions)), 4),
            "pages": total_pages,
            "seconds": round(total_seconds, 2),
            "pages_per_sec": round(total_pages / total_seconds, 3) if total_seconds else 0.0,
        },
    }


def run_ocr_benchmark(
    *,
    corpus_root: Path = EVAL_CORPUS,
    golden_path: Path = GOLDEN_PATH,
    results_dir: Path = RESULTS_DIR,
    ingest: IngestCallable = ingest_corpus,
    verbose: int | None = None,
) -> dict[str, Any]:
    """Benchmark the OCR engines and persist the results (spec §16).

    Args:
        corpus_root: The eval corpus holding the OCR fixtures.
        golden_path: The golden dataset file (its scanned-document queries
            drive the retrieval-impact measurement).
        results_dir: Directory for the timestamped results JSON.
        ingest: Ingest callable (``ingest_corpus``, or the Prefect
            ``ingest_flow`` when invoked through the eval flow).
        verbose: Console verbosity (0–2); defaults to
            ``settings.DEFAULT_VERBOSE``.

    Returns:
        The results document (also written to ``results_dir``), with a
        ``"results_path"`` key naming the file.

    Raises:
        ValueError: If ``verbose`` is invalid or no scanned-document golden
            queries exist.
        FileNotFoundError: If a fixture, ground-truth file, or the golden
            set is missing.
        RuntimeError: If an engine's ingest reports failed files (a partial
            index would silently under-report its retrieval scores).
    """
    verbose = check_verbose(get_settings().DEFAULT_VERBOSE if verbose is None else verbose)
    entries = resolve_golden(load_golden(golden_path), corpus_root)
    scanned = _scanned_entries(entries)
    if not scanned:
        raise ValueError(
            f"no golden queries target the OCR fixtures {sorted(OCR_FIXTURE_PAGES)} — "
            "the retrieval-impact measurement needs at least one"
        )
    embeddings = get_model("embedding")

    engines: dict[str, Any] = {}
    with ephemeral_stores(index_name="varagity_ocr_bench_bm25") as stores:
        retrievers: dict[str, Retriever] = {
            "semantic": SemanticRetriever(store=stores.store, embeddings=embeddings),
            "bm25": BM25Retriever(bm25=stores.bm25, store=stores.store),
            "hybrid": HybridRetriever(store=stores.store, bm25=stores.bm25, embeddings=embeddings),
        }
        for engine in ENGINES:
            # CONTEXTUALIZE off: the engine's text is the only variable
            # (and no LLM nondeterminism between the two engine runs).
            with pinned_eval_settings(OCR_ENGINE=engine, CONTEXTUALIZE="false"):
                intrinsic = _measure_intrinsic(engine, corpus_root, verbose)

                logger.info("[%s] ingesting the fixtures corpus into ephemeral stores", engine)
                started = time.monotonic()
                summary = ingest(
                    str(corpus_root),
                    store=stores.store,
                    bm25=stores.bm25,
                    embeddings=embeddings,
                    reingest=True,  # replace the previous engine's chunks
                    verbose=verbose,
                )
                ingest_seconds = round(time.monotonic() - started, 2)
                if summary.failed:
                    raise RuntimeError(f"[{engine}] ingest failed for {summary.failed} file(s)")
                unresolvable = validate_golden_against_store(scanned, stores, strict=False)

                retrieval: dict[str, Any] = {
                    "ingest_seconds": ingest_seconds,
                    "chunks_ingested": summary.chunks,
                    "unresolvable_golden_refs": unresolvable,
                    "methods": {},
                }
                for method, retriever in retrievers.items():
                    scores, _ranks = measure_retriever(retriever, scanned)
                    retrieval["methods"][method] = scores
                engines[engine] = {"intrinsic": intrinsic, "retrieval": retrieval}

    results: dict[str, Any] = {
        "kind": "ocr_benchmark",
        "timestamp": datetime.now(UTC).isoformat(),
        "corpus": str(corpus_root),
        "golden_path": str(golden_path),
        "fixtures": dict(OCR_FIXTURE_PAGES),
        "n_scanned_queries": len(scanned),
        "k_values": list(K_VALUES),
        "embedding_model": get_settings().EMBEDDING_MODEL,
        "contextualize": False,
        "engines": engines,
    }
    results["results_path"] = str(write_results("ocr", results, results_dir=results_dir))
    return results
