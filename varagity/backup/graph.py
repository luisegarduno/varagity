"""The graph bundle: the engine's working directory, verbatim (ADR-019).

A graph backup is the smallest thing that restores the message graph with no
model call — and for the graph that is the whole working directory. Its
files are the extracted graph itself, its vector indexes, the extracted text
and chunk tracking, the doc-status resume ledger, and the LLM response cache
that a later incremental build rebuilds shared entities from; beside them sit
the repo's two sidecars, the build manifest that keeps the next build a
diffing upsert and the size summary. Nothing is converted or re-derived —
every file goes into the bundle byte for byte — so a bundle is tied to the
engine version that wrote it, which is why its manifest records that
engine's file format (:attr:`~varagity.backup.bundle.BundleManifest.engine`).

The bundle's data members, under one folder named for the engine::

    lightrag/                             the workdir verbatim, *.tmp* skipped   deflated
    ├── kv_store_*.json                   LightRAG's key-value stores (doc status, the LLM cache …)
    ├── vdb_*.json                        its vector indexes
    ├── graph_chunk_entity_relation.graphml
    ├── varagity_manifest.json            the build diff
    └── varagity_graph_summary.json       the size cache — the bundle's counts

**The write lock does the hard part.** :func:`export_graph` copies the files
under :meth:`~varagity.graph.service.GraphService.writing`, the lock every
build, resume and delete takes, so no build can change the graph mid-copy —
while reads take no lock, so chat and ``/graph`` keep answering throughout.
A read does write one file: LightRAG flushes its LLM response cache to disk
after every query. That is harmless, because the engine writes each file to
a ``*.tmp*`` sibling and renames it into place: every member is a whole
file, from before a query or after it, and a cache one query newer than the
rest of the graph is still a valid cache. The ``*.tmp*`` siblings are never
copied.

**The stale flag is read last.** Deleting a graph source flags the graph
stale without taking any lock, so the flag is read once every file is
copied, just before the manifest is sealed: a source deleted mid-export is
then recorded. Only a completed rebuild clears the flag, and the corpus gate
keeps rebuilds out for the whole export, so a late read is never less
accurate than an early one.

**A restore is extract, swap and open** (:func:`restore_graph`), with no
model call. The bundle's files are extracted — each member's sha256 checked
again as it is read — into a staging folder beside the workdir, on the same
volume, and :meth:`~varagity.graph.service.GraphService.replace_workdir`
swaps it in under the service's write and open locks, opens it, and puts
the previous graph back if it won't open. The graph-stale flag the restore
leaves is the bundle's own (the ``graph-restore`` flow writes it).
"""

import logging
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from varagity.backup.bundle import (
    BundleManifest,
    NothingToExport,
    WrittenBundle,
    graph_folder,
    inspect_bundle,
    is_graph_file_name,
    open_bundle,
    write_bundle,
)
from varagity.config import get_settings
from varagity.graph.manifest import load_summary
from varagity.graph.records import GraphStats
from varagity.graph.service import GraphService, GraphUnavailable

logger = logging.getLogger(__name__)

GraphExportStage = Literal["snapshot", "pack", "finalize"]
"""The graph export's progress stages, in order."""

GraphExportProgressCallback = Callable[[GraphExportStage, int | None, int | None], None]
"""``(stage, current, total)`` — how a graph export reports where it is (``pack`` counts bytes)."""

GraphRestoreStage = Literal["validate", "extract", "swap", "open"]
"""The graph restore's progress stages, in order."""

GraphRestoreProgressCallback = Callable[[GraphRestoreStage, int | None, int | None], None]
"""``(stage, current, total)`` — how a graph restore reports progress (``extract`` counts bytes)."""

# The workdir's files are copied into (and out of) a bundle in slices this size.
_COPY_BYTES = 4 << 20

# Progress frames for the byte-counted pack stage: a smooth bar over a few
# hundred megabytes, without flooding the status stream.
_PROGRESS_FRAMES = 50


@dataclass(frozen=True)
class GraphExportReport:
    """What one graph export wrote.

    Attributes:
        bundle: The published bundle (name, path, size, manifest).
        files: How many workdir files it holds.
        wall_clock_s: Seconds from the export's start (the write lock is
            taken at once) to the published bundle.
    """

    bundle: WrittenBundle
    files: int
    wall_clock_s: float

    @property
    def counts(self) -> dict[str, int]:
        """What the bundle holds, by the graph's own summary (see :func:`graph_counts`).

        Returns:
            The manifest's counts.
        """
        return dict(self.bundle.manifest.counts)


def is_temp_file(name: str) -> bool:
    """Report whether a workdir file is an engine's in-flight write.

    LightRAG writes ``<file>.tmp.<pid>.<thread>.<ns>`` beside a file and
    renames it into place; the repo's sidecars use ``<file>.<pid>.tmp``. A
    leftover one is a crash's debris, never part of the graph.

    Args:
        name: A file name in the workdir.

    Returns:
        ``True`` for any name containing ``.tmp`` (the ``*.tmp*`` pattern).
    """
    return ".tmp" in name


def _workdir_entries(workdir: Path) -> tuple[list[Path], list[tuple[str, str]]]:
    """Split a workdir into the files a bundle carries and the entries it can't.

    Args:
        workdir: The engine's working directory.

    Returns:
        The files to pack, in name order, and ``(name, why)`` for every
        other entry an export must skip — a folder, a symlink, a name the
        restore's verdict would refuse. Temp files are dropped silently:
        they are expected, and never part of the graph.
    """
    files: list[Path] = []
    skipped: list[tuple[str, str]] = []
    if not workdir.is_dir():
        return files, skipped
    for entry in sorted(workdir.iterdir(), key=lambda path: path.name):
        name = entry.name
        if is_temp_file(name):
            continue
        if entry.is_symlink():
            skipped.append((name, "a symlink"))
        elif entry.is_dir():
            skipped.append((name, "a folder"))
        elif not entry.is_file():
            skipped.append((name, "not a regular file"))
        elif not is_graph_file_name(name):
            skipped.append((name, "a name a bundle can't carry"))
        else:
            files.append(entry)
    return files, skipped


def workdir_files(workdir: Path) -> list[Path]:
    """List the files a graph export would copy.

    What the export route's ``nothing_to_export`` check asks — the same
    rule the export applies, so the route and the run can never disagree on
    whether there is anything to back up.

    Args:
        workdir: The engine's working directory (a missing one holds none).

    Returns:
        The workdir's own files, in name order, temp files and anything a
        bundle can't carry left out.
    """
    return _workdir_entries(workdir)[0]


def graph_counts(workdir: Path) -> dict[str, int]:
    """What a graph bundle holds, by the workdir's own size summary.

    The summary sidecar is the graph's size cache (what the status route and
    the gauges read) — never the graphml, which a count would have to parse.

    Args:
        workdir: The engine's working directory.

    Returns:
        ``thread_days`` (transcript documents) and ``messages`` (distinct
        guids), plus ``entities`` and ``relations`` when the graph reported
        them — or nothing at all when no build has written a summary yet (a
        first build killed partway leaves files, but no summary).
    """
    summary = load_summary(workdir)
    if summary is None:
        return {}
    counts = {"thread_days": summary.docs, "messages": summary.message_guids}
    if summary.entities is not None:
        counts["entities"] = summary.entities
    if summary.relations is not None:
        counts["relations"] = summary.relations
    return counts


class _ByteProgress:
    """Throttled byte counts for a copying stage — about :data:`_PROGRESS_FRAMES` frames.

    Args:
        report: The export's or restore's progress callback.
        total: The bytes the stage will copy (the files' sizes when listed).
        stage: The stage being counted (an export's ``pack``, a restore's
            ``extract``).
    """

    def __init__(self, report: Callable[..., None], total: int, stage: str = "pack") -> None:
        self._report = report
        self._stage = stage
        self._total = total
        self._step = max(1, total // _PROGRESS_FRAMES)
        self._done = 0
        self._reported = 0
        self._next = self._step
        report(stage, 0, total)

    def advance(self, size: int) -> None:
        """Count copied bytes, reporting when a frame boundary is crossed.

        Args:
            size: Bytes just copied.
        """
        self._done += size
        if self._done >= self._next:
            self._emit()

    def finish(self) -> None:
        """Report the final count, unless the last frame already said it."""
        if self._done != self._reported:
            self._emit()

    def _emit(self) -> None:
        """Send one frame (a file that grew since it was listed never overflows the bar)."""
        self._reported = self._done
        self._next = self._done + self._step
        self._report(self._stage, self._done, max(self._total, self._done))


def export_graph(
    service: GraphService,
    dest: Path,
    *,
    read_stale: Callable[[], bool],
    on_progress: GraphExportProgressCallback | None = None,
) -> GraphExportReport:
    """Export the message graph's working directory into a new bundle in ``dest``.

    Holds the service's write lock for the whole copy — no build, resume or
    delete can start until the bundle is published, while reads (graph chat,
    ``/graph``) are answered as usual. No model is called and no session is
    opened: the files are copied as they are on disk, which is exactly what
    an API restart would load.

    Args:
        service: The process's graph service (its workdir, write lock, and
            engine format).
        dest: The exports folder (``<GRAPH_DOCS_PATH>/varagity-exports``).
        read_stale: Reads the graph-stale flag (the flow passes a Postgres
            read). Called once, after the last file is copied and before the
            manifest is sealed, so a source deleted mid-export is recorded;
            if it raises, the export fails and leaves no bundle.
        on_progress: Called with ``(stage, current, total)`` as the export
            moves through ``snapshot`` (take the lock, list the files) →
            ``pack`` (bytes copied of the files' total) → ``finalize``.

    Returns:
        The published bundle and how long it took.

    Raises:
        varagity.graph.service.GraphBuildInProgress: If a build, resume or
            delete holds the write lock.
        NothingToExport: If the workdir holds no file to copy (nothing has
            been built).
        ValueError: If the engine's name could not be the bundle's folder.
        OSError: If a file can't be read or the bundle can't be written.
    """
    settings = get_settings()
    report: GraphExportProgressCallback = on_progress if on_progress is not None else _ignore
    started = time.perf_counter()
    created_at = datetime.now(UTC)

    report("snapshot", None, None)
    with service.writing():
        files, skipped = _workdir_entries(service.workdir)
        for name, why in skipped:
            logger.warning("not exporting %s from the graph workdir — %s", name, why)
        if not files:
            raise NothingToExport(
                "nothing has been built into the graph — there is nothing to export"
            )
        engine = service.storage_format()
        if not is_graph_file_name(engine.name):
            raise ValueError(f"the graph engine {engine.name!r} can't name a bundle folder")
        counts = graph_counts(service.workdir)
        total = sum(path.stat().st_size for path in files)
        # File names only in these lines: they are relayed into the panel's
        # log, where an in-container path (/app/graph-data/…) would read as
        # nonsense beside the checkout-relative "Saved to" path.
        logger.info(
            "exporting %d graph file(s), %s bytes, written by %s %s",
            len(files),
            f"{total:,}",
            engine.package,
            engine.version,
        )
        with write_bundle(dest, "graph", created_at=created_at) as bundle:
            progress = _ByteProgress(report, total)
            for path in files:
                with (
                    path.open("rb") as handle,
                    bundle.member(f"{engine.name}/{path.name}") as member,
                ):
                    while block := handle.read(_COPY_BYTES):
                        member.write(block)
                        progress.advance(len(block))
            progress.finish()
            report("finalize", None, None)
            # Last, not first: a source deleted during the copy has flagged
            # the graph stale by now (see the module docstring).
            stale = read_stale()
            bundle.seal(
                embedding_model=settings.EMBEDDING_MODEL,
                embedding_dim=settings.EMBEDDING_DIM,
                stale=stale,
                counts=counts,
                engine=engine,
            )
    written = bundle.result
    wall_clock_s = time.perf_counter() - started
    logger.info(
        "exported %d graph file(s) to %s (%s bytes) in %.1fs",
        len(files),
        written.name,
        f"{written.size_bytes:,}",
        wall_clock_s,
    )
    return GraphExportReport(bundle=written, files=len(files), wall_clock_s=wall_clock_s)


# ── restore ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GraphRestore:
    """What one graph restore did.

    Attributes:
        bundle_name: The restored bundle's file name.
        size_bytes: The bundle's size on disk.
        manifest: Its manifest.
        files: How many workdir files it restored.
        stats: The restored graph's size, as its freshly opened session
            reported it.
        wall_clock_s: Seconds from validation to an open graph.
        stale: The graph-stale flag the restore left — the bundle's own,
            written by the ``graph-restore`` flow; ``None`` until it has.
    """

    bundle_name: str
    size_bytes: int
    manifest: BundleManifest
    files: int
    stats: GraphStats
    wall_clock_s: float
    stale: bool | None = None

    @property
    def counts(self) -> dict[str, int]:
        """What the restored graph holds, by the bundle's manifest (its size summary at export).

        Returns:
            ``thread_days``, ``messages`` and — when the graph reported them
            at export — ``entities`` and ``relations``.
        """
        return dict(self.manifest.counts)


def restore_graph(
    service: GraphService,
    bundle: Path,
    *,
    run_id: str,
    on_progress: GraphRestoreProgressCallback | None = None,
) -> GraphRestore:
    """Replace the message graph's working directory with a bundle's.

    No model is called and nothing is re-extracted — the bundle *is* a
    workdir:

    1. ``validate`` — the cheap verdict again (the route verified the
       bundle, every member's sha256 included, before the run; this guards a
       direct call and a file swapped since);
    2. ``extract`` — every file into a fresh staging folder beside the
       workdir (:meth:`~varagity.graph.service.GraphService.staging_dir`, on
       the same volume), each member's sha256 checked again as it is read
       and each file synced to disk, so a bundle that changed after
       verification can't go live;
    3. ``swap`` and ``open`` —
       :meth:`~varagity.graph.service.GraphService.replace_workdir` closes
       the session, swaps the folders under the service's write and open
       locks, and opens the restored graph; if it won't open, the previous
       graph is put back.

    Whatever is left of the staging folder is removed however the restore
    ends; until the swap, the live graph is untouched.

    Args:
        service: The process's graph service (the workdir's single writer).
        bundle: The bundle file.
        run_id: The restore run's id (names the folders beside the workdir).
        on_progress: Called with ``(stage, current, total)`` per stage;
            ``extract`` counts bytes.

    Returns:
        What was restored, and how long it took.

    Raises:
        BundleInvalid: If the bundle is unsound or damaged (the live graph
            is untouched).
        BundleIncompatible: If this stack must not restore it (untouched).
        varagity.graph.service.GraphBuildInProgress: If a write holds the
            graph's write lock at the swap (untouched).
        varagity.graph.service.GraphUnavailable: If the restored graph would
            not open (the previous graph is back in place).
        OSError: If a file can't be extracted or a folder can't be moved
            (the previous graph stays, or is put back).
    """
    report: GraphRestoreProgressCallback = on_progress if on_progress is not None else _ignore
    started = time.perf_counter()

    report("validate", None, None)
    manifest = inspect_bundle(bundle, "graph").manifest_or_raise()
    folder = f"{graph_folder(manifest)}/"
    engine = manifest.engine
    size_bytes = bundle.stat().st_size
    total = sum(member.size_bytes for member in manifest.members)
    # File names only, as the export logs: these lines are relayed into the
    # panel, where an in-container path would read as nonsense.
    logger.info(
        "restoring %d graph file(s), %s bytes, written by %s, from %s",
        len(manifest.members),
        f"{total:,}",
        f"{engine.package} {engine.version}" if engine is not None else "an unknown engine",
        bundle.name,
    )
    staged = service.staging_dir(run_id)
    staged.mkdir(parents=True)  # a fresh folder, never one that already holds files
    try:
        progress = _ByteProgress(report, total, stage="extract")
        with open_bundle(bundle, manifest) as source:
            for member in manifest.members:
                # The verdict held every member to "<folder><flat safe name>".
                target = staged / member.name.removeprefix(folder)
                with source.member(member.name) as stream, target.open("xb") as out:
                    while block := stream.read(_COPY_BYTES):
                        out.write(block)
                        progress.advance(len(block))
                    out.flush()
                    os.fsync(out.fileno())
        progress.finish()
        report("swap", None, None)
        try:
            stats = service.replace_workdir(
                staged, run_id, on_open=lambda: report("open", None, None)
            )
        except GraphUnavailable:
            logger.warning(
                "the restored graph would not open, so the restore was undone — the graph is "
                "as it was before"
            )
            raise
    finally:
        # Gone already after a swap; a failed restore's extracted files otherwise.
        shutil.rmtree(staged, ignore_errors=True)
    wall_clock_s = time.perf_counter() - started
    logger.info(
        "restored %d graph file(s) from %s in %.1fs",
        len(manifest.members),
        bundle.name,
        wall_clock_s,
    )
    return GraphRestore(
        bundle_name=bundle.name,
        size_bytes=size_bytes,
        manifest=manifest,
        files=len(manifest.members),
        stats=stats,
        wall_clock_s=wall_clock_s,
    )


def _ignore(stage: str, current: int | None, total: int | None) -> None:
    """The do-nothing progress callback.

    Args:
        stage: Ignored.
        current: Ignored.
        total: Ignored.
    """
