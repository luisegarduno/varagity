"""The graph workdir's sidecar files — build diff and durable provenance.

Two small JSON files live beside the engine's own storage inside the
workdir, both owned by this module and both written atomically
(tmp + :func:`os.replace`, the discipline the engine's own storages use):

* :data:`MANIFEST_FILENAME` — ``doc_key → {content_sha256, message_guids,
  thread_name, span}`` for every transcript document the graph holds. It
  earns its keep twice (stage-2 decision #9):

  1. **It is the upsert diff LightRAG refuses to do.** The engine's enqueue
     stage drops a re-submitted ``doc_id`` in *any* status, so a re-exported
     ``chat.db`` whose most recent thread-day gained messages would silently
     keep the stale transcript. Comparing rendered content hashes against
     the manifest is what turns that into an explicit
     delete-then-reinsert (:attr:`ManifestDiff.changed`).
  2. **It is a durable provenance index.** Stage 1 rebuilt ``doc_key →
     guids`` inside ``build()`` and lost it on every restart, so a
     re-opened session (or a ``--skip-build`` re-score) could not map a
     citation back to messages. The manifest survives both.

* :data:`SUMMARY_FILENAME` — the last known graph size, refreshed by build
  and delete. It is what :meth:`~varagity.graph.base.GraphSession.stats` and
  the Prometheus gauges read, so neither has to walk the graphml at request
  or scrape time.

Neither file is authoritative over the engine: both are caches of what the
last write did. Anything unreadable, malformed, or written by a different
schema version degrades to "no manifest" — i.e. every rendered document
looks new, which is safe (the engine's own dedup absorbs the re-enqueue)
where trusting a mis-shaped file would not be.
"""

import hashlib
import logging
import os
from collections.abc import Collection, Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from varagity.graph.render import TranscriptDoc

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "varagity_manifest.json"
SUMMARY_FILENAME = "varagity_graph_summary.json"

# Bump when :class:`ManifestDoc` changes shape. A mismatch is read as "no
# manifest": re-deriving costs one build's dedup pass, whereas diffing
# against fields that mean something else costs a wrong graph.
MANIFEST_VERSION = 1


class ManifestDoc(BaseModel):
    """One indexed transcript document, as the workdir remembers it.

    Attributes:
        content_sha256: Hash of the rendered transcript text. Rendering is a
            pure function of the merged messages, so an unchanged
            conversation re-renders to an identical hash no matter how many
            times its source file is re-uploaded.
        message_guids: The document's messages, in transcript order — the
            provenance map behind a document-grain citation.
        thread_name: Human-facing thread label at index time.
        span: The document's day span (``YYYY-MM-DD`` or ``first..last``).
    """

    content_sha256: str
    message_guids: list[str] = []
    thread_name: str = ""
    span: str = ""


class ManifestDiff(BaseModel):
    """What one rendered corpus changes about an indexed workdir.

    Attributes:
        new: Document keys the graph has never seen.
        changed: Keys whose rendered content no longer hashes to what was
            indexed — the engine must **delete** these before re-inserting,
            since its enqueue stage drops an already-known ``doc_id``.
        unchanged: Keys whose content is byte-identical to what is indexed;
            re-inserting them would cost extraction for nothing.
        removed: Keys the workdir holds that this render does not mention.
            Only meaningful for a **full-corpus** render — a bounded build
            (``message_limit`` / ``since``) is deliberately partial, and
            deleting on its say-so would erase the rest of the archive.
    """

    new: list[str] = []
    changed: list[str] = []
    unchanged: list[str] = []
    removed: list[str] = []

    @property
    def pending(self) -> list[str]:
        """Keys that must be (re-)indexed: the new ones and the changed ones.

        Returns:
            ``new + changed``, in that order.
        """
        return [*self.new, *self.changed]


class WorkdirManifest(BaseModel):
    """Every transcript document one workdir's graph was built from.

    Attributes:
        version: Schema version (:data:`MANIFEST_VERSION`).
        docs: ``doc_key`` → what was indexed under it.
    """

    version: int = MANIFEST_VERSION
    docs: dict[str, ManifestDoc] = {}

    def guid_index(self) -> dict[str, list[str]]:
        """Project the manifest into the provenance index the adapters use.

        Returns:
            ``doc_key`` → its message guids, the same shape
            :func:`varagity.graph.render.doc_guid_index` produces from a
            fresh render.
        """
        return {key: list(doc.message_guids) for key, doc in self.docs.items()}

    def thread_name_index(self) -> dict[str, str]:
        """Project the manifest into the display-name map the adapters use.

        A retrieved chunk-grain passage carries no transcript header, so its
        excerpt would otherwise be labelled with the thread *id* while a
        doc-grain hit of the same document shows the rendered display name —
        two labels for one citation target. This index resolves both grains
        to the name the document was indexed under.

        Returns:
            ``doc_key`` → the thread's human-facing label at index time.
            Records without a name are omitted, so a miss falls through to
            the caller's next resolution step instead of blanking the label.
        """
        return {key: doc.thread_name for key, doc in self.docs.items() if doc.thread_name}

    def message_guid_count(self) -> int:
        """Count the distinct messages the manifest accounts for.

        Returns:
            The number of distinct guids across every document (documents
            never overlap in practice, but distinctness is what the number
            claims).
        """
        return len({guid for doc in self.docs.values() for guid in doc.message_guids})

    def diff(self, docs: Sequence[TranscriptDoc]) -> ManifestDiff:
        """Classify a freshly rendered corpus against what is indexed.

        Args:
            docs: The rendered transcripts (see
                :func:`varagity.graph.render.thread_transcripts`).

        Returns:
            The four-way split. Rendered order is preserved within each
            bucket; ``removed`` is sorted, having no render order to inherit.
        """
        diff = ManifestDiff()
        rendered: set[str] = set()
        for doc in docs:
            rendered.add(doc.doc_key)
            known = self.docs.get(doc.doc_key)
            if known is None:
                diff.new.append(doc.doc_key)
            elif known.content_sha256 == content_sha256(doc.text):
                diff.unchanged.append(doc.doc_key)
            else:
                diff.changed.append(doc.doc_key)
        diff.removed = sorted(key for key in self.docs if key not in rendered)
        return diff

    def merged(
        self,
        docs: Sequence[TranscriptDoc],
        *,
        prune: bool,
        retain: Collection[str] = (),
    ) -> "WorkdirManifest":
        """Fold a rendered corpus into a new manifest.

        Args:
            docs: The rendered transcripts the build just indexed.
            prune: Whether the render was the **whole** corpus. ``True``
                drops documents this render did not mention (their source
                messages are gone); ``False`` keeps them, which is what a
                bounded build needs — its render is partial by construction.
            retain: Keys whose **existing** record must survive verbatim,
                for documents the engine refused to delete. Their old
                content is still in the graph, so writing the new hash would
                make the next build believe they are up to date and the
                stale transcript would become permanent — the exact failure
                this manifest exists to prevent.

        Returns:
            The new manifest (this one is left untouched).
        """
        entries = {} if prune else dict(self.docs)
        for doc in docs:
            if doc.doc_key in retain:
                continue
            entries[doc.doc_key] = ManifestDoc(
                content_sha256=content_sha256(doc.text),
                message_guids=list(doc.message_guids),
                thread_name=doc.thread_name,
                span=doc.doc_key.partition("::")[2],
            )
        for key in retain:
            if (known := self.docs.get(key)) is not None:
                entries[key] = known
        return WorkdirManifest(version=MANIFEST_VERSION, docs=entries)

    def without(self, doc_keys: Sequence[str]) -> "WorkdirManifest":
        """Drop documents that were deleted from the graph.

        Args:
            doc_keys: The keys removed from the engine.

        Returns:
            The new manifest (this one is left untouched).
        """
        dropped = set(doc_keys)
        return WorkdirManifest(
            version=MANIFEST_VERSION,
            docs={key: doc for key, doc in self.docs.items() if key not in dropped},
        )


class WorkdirSummary(BaseModel):
    """The last known size of one workdir's graph.

    A cache, not a source of truth: it is written by whatever last changed
    the graph, so a workdir mutated behind the adapter's back reports what
    the adapter last saw. That is the deliberate trade — Prometheus scrapes
    and status polls must never parse a multi-megabyte graphml.

    Attributes:
        entities: Node count, or ``None`` when the graph would not say.
        relations: Edge count, or ``None`` when the graph would not say.
        message_guids: Messages the manifest accounts for.
        docs: Transcript documents the manifest accounts for.
        refreshed_at: When this summary was written (aware UTC) — the
            staleness signal for anything reading it.
    """

    entities: int | None = None
    relations: int | None = None
    message_guids: int = 0
    docs: int = 0
    refreshed_at: datetime


def content_sha256(text: str) -> str:
    """Hash a rendered transcript for the build diff.

    Args:
        text: The rendered document text.

    Returns:
        The hex digest of its UTF-8 bytes.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def manifest_path(workdir: Path) -> Path:
    """Locate the manifest inside a workdir.

    Args:
        workdir: The engine's working directory.

    Returns:
        The manifest's path (which may not exist yet).
    """
    return workdir / MANIFEST_FILENAME


def summary_path(workdir: Path) -> Path:
    """Locate the summary sidecar inside a workdir.

    Args:
        workdir: The engine's working directory.

    Returns:
        The sidecar's path (which may not exist yet).
    """
    return workdir / SUMMARY_FILENAME


def load_manifest(workdir: Path) -> WorkdirManifest:
    """Read a workdir's manifest, degrading to an empty one.

    A missing, unreadable, malformed, or differently-versioned file yields
    an empty manifest: every rendered document then looks new, the engine's
    own dedup absorbs the re-enqueue, and the next build writes a correct
    file. The alternative — trusting a file whose fields may mean something
    else — would corrupt the graph silently.

    Args:
        workdir: The engine's working directory.

    Returns:
        The manifest, or an empty one.
    """
    path = manifest_path(workdir)
    try:
        manifest = WorkdirManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return WorkdirManifest()
    except (OSError, ValueError):
        logger.warning("graph manifest at %s is unreadable — treating it as empty", path)
        return WorkdirManifest()
    if manifest.version != MANIFEST_VERSION:
        logger.warning(
            "graph manifest at %s is version %d, expected %d — treating it as empty",
            path,
            manifest.version,
            MANIFEST_VERSION,
        )
        return WorkdirManifest()
    return manifest


def save_manifest(workdir: Path, manifest: WorkdirManifest) -> None:
    """Write a workdir's manifest atomically.

    Args:
        workdir: The engine's working directory (created if absent).
        manifest: The manifest to persist.
    """
    _write_atomic(manifest_path(workdir), manifest.model_dump_json(indent=2) + "\n")


def load_summary(workdir: Path) -> WorkdirSummary | None:
    """Read a workdir's summary sidecar.

    Args:
        workdir: The engine's working directory.

    Returns:
        The summary, or ``None`` when there is none to read (which callers
        answer by asking the graph itself).
    """
    path = summary_path(workdir)
    try:
        return WorkdirSummary.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        logger.warning("graph summary at %s is unreadable — ignoring it", path)
        return None


def save_summary(
    workdir: Path,
    manifest: WorkdirManifest,
    *,
    entities: int | None,
    relations: int | None,
) -> WorkdirSummary:
    """Refresh a workdir's summary sidecar atomically.

    Args:
        workdir: The engine's working directory (created if absent).
        manifest: The manifest the document/message counts come from.
        entities: Node count from the graph, or ``None`` if it would not say.
        relations: Edge count from the graph, or ``None`` if it would not say.

    Returns:
        The summary that was written.
    """
    summary = WorkdirSummary(
        entities=entities,
        relations=relations,
        message_guids=manifest.message_guid_count(),
        docs=len(manifest.docs),
        refreshed_at=datetime.now(UTC),
    )
    _write_atomic(summary_path(workdir), summary.model_dump_json(indent=2) + "\n")
    return summary


def _write_atomic(path: Path, payload: str) -> None:
    """Replace a file's contents in one step, or leave the old one intact.

    A half-written manifest is worse than a missing one — it would diff a
    real corpus against a truncated record — so the write lands in a
    temporary file beside the target and is moved over it, which is atomic
    within a filesystem.

    Args:
        path: The file to write.
        payload: Its new contents.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        logger.warning("could not write %s", path, exc_info=True)
