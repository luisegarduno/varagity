"""Prefect evaluation flows: tracked runs of the spec §16 harness.

Thin ``@flow`` shells over :func:`varagity.eval.evaluate.run_matrix`,
:func:`varagity.eval.evaluate.run_chat_eval` and
:func:`varagity.eval.ocr_benchmark.run_ocr_benchmark`, passing the tracked
:func:`~varagity.pipeline.ingest_flow.ingest_flow` through the harness's
ingest seam — every eval ingest (two for the matrix, one each for the
chat eval and per OCR engine) is a subflow with per-stage task runs at
the Prefect UI, and the eval run itself is one flow run.

Like the other flows, parameter validation is off (duck-typed internals)
and the measurement work carries no Prefect-level retries: an unreachable
GPU service or Docker daemon should fail the run loudly, not back off —
the clients already retry transient HTTP failures internally.
"""

from typing import Any

from prefect import flow

from varagity.eval.evaluate import run_chat_eval, run_matrix
from varagity.eval.ocr_benchmark import run_ocr_benchmark
from varagity.pipeline.ingest_flow import ingest_flow


@flow(name="eval-matrix", validate_parameters=False)
def eval_flow(verbose: int | None = None) -> dict[str, Any]:
    """Run the 7-configuration retrieval matrix as a tracked flow run.

    Args:
        verbose: Console verbosity (0–2); defaults to
            ``settings.DEFAULT_VERBOSE``.

    Returns:
        The results document (see
        :func:`varagity.eval.evaluate.run_matrix`).
    """
    return run_matrix(ingest=ingest_flow, verbose=verbose)


@flow(name="eval-chat", validate_parameters=False)
def chat_eval_flow(verbose: int | None = None) -> dict[str, Any]:
    """Run the multi-turn chat-engine eval as a tracked flow run.

    Args:
        verbose: Console verbosity (0–2); defaults to
            ``settings.DEFAULT_VERBOSE``.

    Returns:
        The results document (see
        :func:`varagity.eval.evaluate.run_chat_eval`).
    """
    return run_chat_eval(ingest=ingest_flow, verbose=verbose)


@flow(name="eval-ocr", validate_parameters=False)
def ocr_benchmark_flow(verbose: int | None = None) -> dict[str, Any]:
    """Run the OCR engine benchmark as a tracked flow run.

    Args:
        verbose: Console verbosity (0–2); defaults to
            ``settings.DEFAULT_VERBOSE``.

    Returns:
        The results document (see
        :func:`varagity.eval.ocr_benchmark.run_ocr_benchmark`).
    """
    return run_ocr_benchmark(ingest=ingest_flow, verbose=verbose)
