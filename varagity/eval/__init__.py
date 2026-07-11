"""Offline evaluation harness (spec §16, Phase 9).

Measures retrieval quality — recall@k / pass@k over a hand-authored golden
set — across the four Contextual Retrieval configurations, and benchmarks
the pluggable OCR engines. Runs against ephemeral testcontainers stores
(plan decision #4) with the live GPU services for embeddings/LLM.

Heavy eval-only dependencies (``testcontainers``, ``jiwer`` — the ``eval``
dependency group) are imported at call time, so importing this package
never requires them.
"""

from varagity.eval.datasets import GoldenEntry, load_golden, resolve_golden
from varagity.eval.evaluate import pass_at_k, recall_at_k, run_matrix
from varagity.eval.ocr_benchmark import run_ocr_benchmark

__all__ = [
    "GoldenEntry",
    "load_golden",
    "pass_at_k",
    "recall_at_k",
    "resolve_golden",
    "run_matrix",
    "run_ocr_benchmark",
]
