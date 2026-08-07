"""Prefect graph flows: the graph build as a tracked run (spec_graphrag §5.2).

A thin ``@flow`` shell over :meth:`varagity.graph.service.GraphService.build`,
for the same reason :mod:`varagity.pipeline.eval_flow` is one: the work
itself belongs to the service (which owns the session, the write lock, and
the manifest diff), while Prefect owns the *record* that a build ran, how
long it took, and whether it failed. A multi-day backfill with no run in the
UI would be an invisible job.

Deliberately **one flow run per build attempt, not per document**: the engine
enqueues documents into its own durable status store and processes them in
batches inside a single call, so per-document task runs would have to reach
inside the engine's pipeline. Per-document progress is reported instead by
the runner's status sampler (:mod:`varagity.api.graph_runner`), which reads
the engine's own document counts — tracking and streaming compose, as they
do for ingest.

Parameter validation is off (the service is a live handle, not a
serializable payload) and there are no Prefect-level retries: an engine that
fails mid-backfill must surface, not silently re-run hours of extraction —
and a re-called build resumes from the durable statuses anyway.
"""

from collections.abc import Sequence

from prefect import flow
from prefect.logging import get_run_logger

from varagity.graph.records import BuildReport
from varagity.graph.service import GraphService
from varagity.graph.sources.base import MessageBatch


@flow(name="graph-build", validate_parameters=False)
def graph_build_flow(
    service: GraphService,
    batches: Sequence[MessageBatch],
    *,
    prune_removed: bool = True,
    verbose: int = 0,
) -> BuildReport:
    """Upsert a parsed message corpus into the graph as a tracked flow run.

    Args:
        service: The process's graph service (holds the session and the
            single-flight write lock).
        batches: Parsed source files, guid-merged by the session before
            rendering.
        prune_removed: Whether ``batches`` render the whole corpus. A bounded
            build passes ``False`` — its render is partial on purpose, and
            pruning on its say-so would delete the rest of the archive.
        verbose: Validated console verbosity (0–2).

    Returns:
        What the build did (messages seen, wall clock, caught failures).
    """
    logger = get_run_logger()
    logger.info(
        "graph build starting: %d source file(s), %s render",
        len(batches),
        "full-corpus" if prune_removed else "bounded",
    )
    report = service.build(batches, verbose=verbose, prune_removed=prune_removed)
    logger.info(
        "graph build finished: %d message(s) in %.1fs, %d failure(s)",
        report.messages_seen,
        report.wall_clock_s,
        len(report.failures),
    )
    return report
