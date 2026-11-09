"""The backup bundle: one zip per backup, its manifest written last (ADR-019).

A bundle is an ordinary ``.zip`` in a corpus root's reserved
``varagity-exports/`` folder. Its data members stream in first, each hashed
as it is written, and ``bundle.json`` — the :class:`BundleManifest` — is the
**last** member, so it can record every data member's size and sha256. The
zip's central directory then lets :func:`read_manifest` open that one member
without touching the (possibly hundreds of megabytes of) rest: listing the
backups panel costs a directory read, not an unpack.

Writing never exposes a half-made file. :func:`write_bundle` writes into a
hidden ``.{name}.partial`` sibling and renames it into place
(``os.replace``) only once the manifest is sealed and the bytes are synced;
any failure removes the partial instead, and the next export sweeps whatever
a crash left behind.
Listings skip hidden files, so a bundle is either whole or invisible.

Reading has two depths, because hashing is the expensive part:

* :func:`inspect_bundle` is the **cheap verdict** every listed bundle gets —
  the zip's central directory and ``bundle.json`` alone: the format version,
  the corpus, the member allowlist and safety rules, each member's recorded
  size against the zip's, and the embedding model and dimension against this
  stack's. No data member is opened, so a folder of large bundles still lists
  in milliseconds.
* :func:`verify_bundle` adds the sha256 of every member — what a restore
  runs before it touches a store, and the only check that sees a bundle
  damaged *inside* an intact zip. :func:`open_bundle` hands out the same
  verifying streams to the restore itself, so the bytes it loads are checked
  again as they are read.

Two refusals come out of both: :class:`BundleInvalid` (not a sound bundle;
:class:`BundleDamaged` when only its bytes are wrong) and
:class:`BundleIncompatible` (a sound bundle this stack must not restore — a
mismatch is refused, never converted).

This module knows the container, not the corpora: what goes *inside* a bundle
is :mod:`varagity.backup.rag`'s business (and, later, the graph's) — except
each corpus's member allowlist, which the verdict needs — and nothing here
knows about HTTP.
"""

import hashlib
import json
import logging
import os
import re
import stat
import zipfile
import zlib
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any, Literal

from pydantic import BaseModel, ValidationError

from varagity.config import get_settings
from varagity.paths import EXPORTS_DIRNAME, resolve_contained

logger = logging.getLogger(__name__)

Corpus = Literal["rag", "graph"]
"""Which corpus a bundle backs up — ``ChatRequest.corpus``'s vocabulary."""

BUNDLE_FORMAT: Literal["varagity-backup"] = "varagity-backup"
"""The manifest's ``format`` marker: what makes a zip a varagity bundle."""

BUNDLE_VERSION = 1
"""The bundle layout this code writes (and, in time, the versions it reads)."""

MANIFEST_NAME = "bundle.json"
"""The manifest member — always the zip's last entry."""

BUNDLE_SUFFIX = ".zip"
PARTIAL_SUFFIX = ".partial"

# A manifest is a few kilobytes; anything near this is not one (and reading
# a declared-huge member into memory is how a hostile zip would hurt us).
_MAX_MANIFEST_BYTES = 1 << 20

# A data member's zip entry: a regular file, rw-r--r--, so an extracted
# bundle never carries a symlink or an executable bit.
_MEMBER_MODE = (stat.S_IFREG | 0o644) << 16

# Everything zipfile raises for a file that isn't a sound zip: a bad or
# truncated archive, a corrupt deflate stream, an unsupported compression
# method, an encrypted member (RuntimeError), or the filesystem itself.
_UNREADABLE_ZIP = (
    zipfile.BadZipFile,
    OSError,
    EOFError,
    zlib.error,
    NotImplementedError,
    RuntimeError,
)

# What reading a member's *bytes* raises when they are wrong: a CRC-32
# mismatch (BadZipFile), a corrupt or truncated deflate stream. Damage, as
# opposed to a file that was never a sound zip.
_DAMAGED_MEMBER = (zipfile.BadZipFile, EOFError, zlib.error)

# Members are hashed (and drained) in slices this size.
_READ_BYTES = 4 << 20

# A Windows drive prefix ("C:"), refused as an absolute member path.
_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:")

# What a bundle of each corpus holds besides its manifest: the restore's
# member allowlist, matched exactly. The names are varagity.backup.rag's
# member constants, hardcoded again here because that module imports this
# one — a unit test pins the two together (the repo's tuple↔registry
# convention). A corpus missing here cannot be restored at all yet.
_CORPUS_MEMBERS: dict[str, frozenset[str]] = {
    "rag": frozenset({"documents.jsonl", "chunks.jsonl", "embeddings.f32"}),
}

# How a verdict names each corpus to a person.
_CORPUS_NOUNS: dict[str, str] = {"rag": "document corpus", "graph": "message graph"}


class BundleMember(BaseModel):
    """One data member of a bundle, as its manifest records it.

    Attributes:
        name: The member's path inside the zip.
        size_bytes: Its uncompressed size.
        sha256: The hex sha256 of its uncompressed bytes — what a restore
            verifies before it touches a store.
    """

    name: str
    size_bytes: int
    sha256: str


class BundleManifest(BaseModel):
    """``bundle.json``: what a bundle holds and what it was made under.

    Every later phase judges bundles against this record — a restore checks
    compatibility against it and derives the stale flag from it — so it
    carries the export-time facts no row records.

    Attributes:
        format: Always ``"varagity-backup"``.
        version: The bundle layout version (:data:`BUNDLE_VERSION`).
        corpus: Which corpus the bundle backs up.
        created_at: When the export's snapshot was taken (UTC).
        embedding_model: ``EMBEDDING_MODEL`` at export — the store checks
            only vector size, so the model name is the only guard against
            restoring vectors from another embedding space.
        embedding_dim: ``EMBEDDING_DIM`` at export.
        stale: The corpus's stale flag at export.
        counts: What the bundle holds (rag: ``documents``, ``chunks``).
        ingest_settings: The reingest-affecting settings at export, in
            env-string form (rag only; ``None`` for a corpus without them).
        members: Every data member with its size and sha256, in write order.
    """

    format: Literal["varagity-backup"]
    version: int
    corpus: Corpus
    created_at: datetime
    embedding_model: str
    embedding_dim: int
    stale: bool
    counts: dict[str, int]
    ingest_settings: dict[str, str] | None = None
    members: list[BundleMember]


class BundleError(Exception):
    """Base class for a bundle that can't be read or used."""


class BundleInvalid(BundleError):
    """The file is not a readable varagity bundle (damaged, or not one at all)."""


class BundleDamaged(BundleInvalid):
    """A member's bytes don't match what its bundle recorded.

    The one defect the cheap verdict can't see: the zip and its manifest are
    intact, but a member's sha256 (or the zip's own CRC-32, or its deflate
    stream) says the content changed after it was written.
    """


class BundleIncompatible(BundleError):
    """A sound bundle this stack must not restore.

    Another bundle format version, another corpus, or vectors from another
    embedding model or dimension. Neither store checks the embedding model
    (pgvector compares only the vector size), so restoring a mismatch would
    succeed and then retrieve the wrong passages — it is refused instead,
    never converted.
    """


@dataclass(frozen=True)
class BundleVerdict:
    """What the cheap inspection concluded about one bundle.

    Attributes:
        manifest: The bundle's manifest, or ``None`` when none could be read
            (also for a format version this stack can't parse).
        problem: Why the bundle can't be restored, or ``None`` when it can.
    """

    manifest: BundleManifest | None
    problem: BundleError | None = None

    @property
    def restorable(self) -> bool:
        """Whether nothing the inspection could see refuses the bundle.

        Returns:
            ``True`` when there is no problem (the restore still hashes it).
        """
        return self.problem is None

    @property
    def reason(self) -> str | None:
        """The problem, worded for a person.

        Returns:
            The refusal's message, or ``None`` for a restorable bundle.
        """
        return None if self.problem is None else str(self.problem)

    def manifest_or_raise(self) -> BundleManifest:
        """Return the manifest of a restorable bundle, or raise its problem.

        Returns:
            The manifest.

        Raises:
            BundleInvalid: If the bundle is not a sound bundle.
            BundleIncompatible: If this stack must not restore it.
        """
        if self.problem is not None:
            raise self.problem
        if self.manifest is None:  # pragma: no cover — a verdict always pairs one or the other
            raise BundleInvalid("the bundle has no readable manifest")
        return self.manifest


@dataclass(frozen=True)
class WrittenBundle:
    """A bundle :func:`write_bundle` published.

    Attributes:
        name: The bundle's file name.
        path: Where it now lives.
        size_bytes: Its size on disk.
        manifest: The manifest sealed into it.
    """

    name: str
    path: Path
    size_bytes: int
    manifest: BundleManifest


@dataclass(frozen=True)
class BundleListing:
    """One bundle file found in an exports folder.

    Attributes:
        name: The file name.
        path: Its resolved path.
        size_bytes: Its size on disk.
        modified_at: Its mtime (UTC).
        mtime_ns: Its mtime in nanoseconds — with ``size_bytes``, the
            identity a remembered restore failure is keyed by.
        verdict: The cheap inspection's verdict for the listing's corpus.
    """

    name: str
    path: Path
    size_bytes: int
    modified_at: datetime
    mtime_ns: int
    verdict: BundleVerdict

    @property
    def manifest(self) -> BundleManifest | None:
        """The bundle's manifest.

        Returns:
            The manifest, or ``None`` when the file isn't a readable
            varagity bundle (copied in by hand, damaged, another format …).
        """
        return self.verdict.manifest


def corpus_root(corpus: Corpus) -> Path:
    """The configured root directory of one corpus.

    Args:
        corpus: The corpus.

    Returns:
        ``DOCS_PATH`` for ``rag``, ``GRAPH_DOCS_PATH`` for ``graph``.
    """
    settings = get_settings()
    return Path(settings.DOCS_PATH if corpus == "rag" else settings.GRAPH_DOCS_PATH)


def exports_dir(corpus: Corpus) -> Path:
    """The reserved folder a corpus's bundles live in.

    Args:
        corpus: The corpus.

    Returns:
        ``<corpus root>/varagity-exports`` (it may not exist yet).
    """
    return corpus_root(corpus) / EXPORTS_DIRNAME


def display_path(corpus: Corpus, name: str | None = None) -> str:
    """The human form of a bundle's location — the text "Copy path" copies.

    In compose the API sees ``/app/docs`` while the user's checkout has
    ``./docs`` (the bind's host side), so an absolute path would be wrong on
    the one machine that matters. The corpus folder's own name, relative,
    reads right from the checkout root in both host and container mode.

    Args:
        corpus: The corpus.
        name: A bundle file name; ``None`` names the folder itself.

    Returns:
        E.g. ``"docs/varagity-exports/varagity-rag-20261108-1312.zip"`` or
        ``"docs/varagity-exports/"``.
    """
    # A root with no name is "/" itself — and "/varagity-exports/" is then
    # exactly right.
    folder = f"{corpus_root(corpus).resolve().name}/{EXPORTS_DIRNAME}/"
    return folder if name is None else f"{folder}{name}"


def is_bundle_name(name: str) -> bool:
    """Report whether a file name is a listable, servable bundle name.

    Args:
        name: A bare file name (no directories).

    Returns:
        ``True`` for a visible ``*.zip`` name — never a hidden file, so an
        in-flight ``.{name}.partial`` (or anything else dot-prefixed) is
        never listed or served.
    """
    return (
        bool(name)
        and not name.startswith(".")
        and "/" not in name
        and "\\" not in name
        and name.lower().endswith(BUNDLE_SUFFIX)
    )


def bundle_name(directory: Path, corpus: Corpus, *, now: datetime | None = None) -> str:
    """Choose the file name for a new bundle — never one already taken.

    ``varagity-<corpus>-<UTC yyyymmdd-HHMM>.zip``; a repeat within the same
    minute gets a ``-2``, ``-3``, … suffix rather than replacing a backup.

    Args:
        directory: The exports folder the bundle will land in.
        corpus: The corpus being backed up.
        now: The export's instant (defaults to now); rendered in UTC.

    Returns:
        A name free in ``directory`` (also free of an in-flight partial).
    """
    stamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%d-%H%M")
    base = f"varagity-{corpus}-{stamp}"
    candidate = f"{base}{BUNDLE_SUFFIX}"
    attempt = 2
    while _taken(directory, candidate):
        candidate = f"{base}-{attempt}{BUNDLE_SUFFIX}"
        attempt += 1
    return candidate


def _taken(directory: Path, name: str) -> bool:
    """Report whether a bundle name is occupied (by a file or a partial).

    Args:
        directory: The exports folder.
        name: The candidate bundle name.

    Returns:
        ``True`` when anything — even a dangling symlink — holds the name or
        its partial's name.
    """
    return (directory / name).exists(follow_symlinks=False) or (
        directory / f".{name}{PARTIAL_SUFFIX}"
    ).exists(follow_symlinks=False)


def sweep_partials(directory: Path, corpus: Corpus) -> int:
    """Remove partial bundles an interrupted export of ``corpus`` left behind.

    Safe because only one export per corpus runs at a time (the corpus gate),
    and only this corpus's prefix is touched — another corpus's export into
    the same folder is never disturbed.

    Args:
        directory: The exports folder.
        corpus: The corpus whose partials to sweep.

    Returns:
        How many partials were removed.
    """
    removed = 0
    if not directory.is_dir():
        return removed
    for path in directory.glob(f".varagity-{corpus}-*{BUNDLE_SUFFIX}{PARTIAL_SUFFIX}"):
        try:
            path.unlink()
        except OSError as error:
            logger.warning("could not remove the stale partial %s: %s", path.name, error)
            continue
        logger.info("removed %s, left behind by an interrupted export", path.name)
        removed += 1
    return removed


class MemberSink:
    """The write end of one bundle member, hashing and counting as it goes.

    Attributes:
        size_bytes: Bytes written so far (uncompressed).
    """

    def __init__(self, raw: IO[bytes]) -> None:
        """Wrap a zip member's write stream.

        Args:
            raw: The stream ``ZipFile.open(..., "w")`` returned.
        """
        self._raw = raw
        self._hash = hashlib.sha256()
        self.size_bytes = 0

    def write(self, data: bytes | bytearray | memoryview) -> int:
        """Append bytes to the member.

        Args:
            data: The next slice of the member's content.

        Returns:
            The number of bytes written.
        """
        self._raw.write(data)
        self._hash.update(data)
        size = len(data)
        self.size_bytes += size
        return size

    def hexdigest(self) -> str:
        """The sha256 of everything written so far.

        Returns:
            The hex digest.
        """
        return self._hash.hexdigest()


class BundleWriter:
    """Fills one bundle: data members first, then the sealed manifest.

    Obtained from :func:`write_bundle`; :attr:`result` is available once its
    ``with`` block exits cleanly.

    Attributes:
        corpus: The corpus being backed up.
        created_at: The export's instant — the name's stamp and the
            manifest's ``created_at``.
    """

    def __init__(self, archive: zipfile.ZipFile, corpus: Corpus, created_at: datetime) -> None:
        """Bind the writer to its open partial archive.

        Args:
            archive: The partial zip, open for writing.
            corpus: The corpus being backed up.
            created_at: The export's instant (UTC).
        """
        self._archive = archive
        self.corpus = corpus
        self.created_at = created_at
        self._members: list[BundleMember] = []
        self._manifest: BundleManifest | None = None
        self._result: WrittenBundle | None = None
        # Zip stores a naive DOS timestamp; the UTC wall clock is the honest
        # one (the manifest carries the aware instant regardless).
        self._date_time = created_at.astimezone(UTC).timetuple()[:6]

    @contextmanager
    def member(self, name: str, *, compress: bool = True) -> Iterator[MemberSink]:
        """Stream one data member into the bundle.

        Args:
            name: The member's path inside the zip.
            compress: Deflate it (text) or store it as-is (already-dense
                bytes like a float32 matrix, which deflate can't shrink).

        Yields:
            The sink to write the member's bytes to; its size and sha256 are
            recorded for the manifest when the block exits.

        Raises:
            ValueError: If the name is the manifest's or already used.
            RuntimeError: If the manifest is already sealed.
        """
        if self._manifest is not None:
            raise RuntimeError("the manifest is sealed — no member may follow it")
        if name == MANIFEST_NAME or any(member.name == name for member in self._members):
            raise ValueError(f"bundle member {name!r} is reserved or already written")
        info = zipfile.ZipInfo(name, date_time=self._date_time)
        info.compress_type = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
        info.external_attr = _MEMBER_MODE
        # force_zip64: the size is unknown up front, and a corpus's vectors
        # can pass the 2 GiB a plain entry can describe.
        with self._archive.open(info, mode="w", force_zip64=True) as raw:
            sink = MemberSink(raw)
            yield sink
        self._members.append(
            BundleMember(name=name, size_bytes=sink.size_bytes, sha256=sink.hexdigest())
        )

    def seal(
        self,
        *,
        embedding_model: str,
        embedding_dim: int,
        stale: bool,
        counts: dict[str, int],
        ingest_settings: dict[str, str] | None = None,
    ) -> BundleManifest:
        """Write ``bundle.json`` as the bundle's last member.

        Args:
            embedding_model: ``EMBEDDING_MODEL`` at export.
            embedding_dim: ``EMBEDDING_DIM`` at export.
            stale: The corpus's stale flag at export.
            counts: What the bundle holds.
            ingest_settings: The reingest-affecting settings at export, if
                the corpus has them.

        Returns:
            The manifest, as written.

        Raises:
            RuntimeError: If the manifest is already sealed.
        """
        if self._manifest is not None:
            raise RuntimeError("the manifest is already sealed")
        manifest = BundleManifest(
            format=BUNDLE_FORMAT,
            version=BUNDLE_VERSION,
            corpus=self.corpus,
            created_at=self.created_at,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            stale=stale,
            counts=dict(counts),
            ingest_settings=dict(ingest_settings) if ingest_settings is not None else None,
            members=list(self._members),
        )
        info = zipfile.ZipInfo(MANIFEST_NAME, date_time=self._date_time)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = _MEMBER_MODE
        self._archive.writestr(info, manifest.model_dump_json(indent=2) + "\n")
        self._manifest = manifest
        return manifest

    @property
    def manifest(self) -> BundleManifest | None:
        """The sealed manifest (``None`` until :meth:`seal` runs).

        Returns:
            The manifest, or ``None``.
        """
        return self._manifest

    @property
    def result(self) -> WrittenBundle:
        """The published bundle.

        Returns:
            What :func:`write_bundle` moved into place.

        Raises:
            RuntimeError: If the bundle was not (yet) published.
        """
        if self._result is None:
            raise RuntimeError("the bundle has not been published")
        return self._result

    def _publish(self, result: WrittenBundle) -> None:
        """Record the published bundle (called by :func:`write_bundle`).

        Args:
            result: The bundle now in place.
        """
        self._result = result


@contextmanager
def write_bundle(
    directory: Path, corpus: Corpus, *, created_at: datetime | None = None
) -> Iterator[BundleWriter]:
    """Write a new bundle into ``directory``, all-or-nothing.

    Sweeps this corpus's stale partials, chooses a free name
    (:func:`bundle_name`), and opens ``.{name}.partial``. When the ``with``
    block exits cleanly *and* the writer was sealed, the partial is synced
    and renamed into place with ``os.replace`` — same directory, so the
    rename is atomic and a listing sees the whole bundle or nothing. Any
    exception, or an unsealed writer, deletes the partial instead.

    Args:
        directory: The exports folder (created if missing).
        corpus: The corpus being backed up.
        created_at: The export's instant (defaults to now).

    Yields:
        The writer to fill; read :attr:`BundleWriter.result` after the block.

    Raises:
        RuntimeError: If the block exits without sealing the manifest (the
            partial is removed; nothing is published).
        OSError: If the folder or the bundle can't be written.
    """
    created = (created_at or datetime.now(UTC)).astimezone(UTC)
    directory.mkdir(parents=True, exist_ok=True)
    sweep_partials(directory, corpus)
    name = bundle_name(directory, corpus, now=created)
    partial = directory / f".{name}{PARTIAL_SUFFIX}"
    try:
        # "x": never clobber a partial some other writer owns.
        with zipfile.ZipFile(partial, mode="x", compression=zipfile.ZIP_DEFLATED) as archive:
            writer = BundleWriter(archive, corpus, created)
            yield writer
            manifest = writer.manifest
            if manifest is None:
                raise RuntimeError(
                    "the bundle was never sealed — a bundle without its manifest is not published"
                )
        _fsync(partial)
        final = directory / name
        if final.exists(follow_symlinks=False):
            # Something claimed the name while we wrote — never replace it.
            final = directory / bundle_name(directory, corpus, now=created)
        os.replace(partial, final)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    _fsync_directory(directory)
    size = final.stat().st_size
    writer._publish(WrittenBundle(name=final.name, path=final, size_bytes=size, manifest=manifest))
    # DEBUG: the corpus export logs its own one-line summary at INFO.
    logger.debug("published %s (%d bytes)", final.name, size)


def _fsync(path: Path) -> None:
    """Flush a finished file to disk before it is renamed into place.

    A backup's whole job is surviving the bad day; a rename that lands
    before the bytes do would publish an empty bundle after a power loss.

    Args:
        path: The file to sync.
    """
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(directory: Path) -> None:
    """Make a rename durable by syncing its directory (best-effort).

    Args:
        directory: The directory the rename happened in.
    """
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _unreadable(error: BaseException) -> BundleInvalid:
    """The refusal for a file zipfile can't open at all.

    Args:
        error: What zipfile (or the filesystem) raised.

    Returns:
        The ``BundleInvalid`` to raise or record.
    """
    return BundleInvalid(f"Not a readable zip file ({error}).")


def _manifest_bytes(archive: zipfile.ZipFile) -> bytes:
    """Read the raw ``bundle.json`` member — the only member this opens.

    Args:
        archive: The open bundle.

    Returns:
        The manifest's bytes.

    Raises:
        BundleInvalid: If there is no manifest, it is implausibly large, or
            its bytes can't be read.
    """
    try:
        info = archive.getinfo(MANIFEST_NAME)
    except KeyError:
        raise BundleInvalid(f"No {MANIFEST_NAME} — not a Varagity backup.") from None
    if info.file_size > _MAX_MANIFEST_BYTES:
        raise BundleInvalid(f"Its {MANIFEST_NAME} is {info.file_size:,} bytes — not a manifest.")
    try:
        return archive.read(info)
    except _UNREADABLE_ZIP as error:
        raise _unreadable(error) from error


def _parse_manifest(raw: bytes) -> BundleManifest:
    """Validate a manifest, telling another format version from a bad file.

    The version is judged *before* the full schema: a bundle written by a
    later layout may not validate as this one, and "another version" is the
    honest verdict for it, not "not a manifest".

    Args:
        raw: The ``bundle.json`` bytes.

    Returns:
        The validated manifest.

    Raises:
        BundleIncompatible: If it is a varagity manifest of another version.
        BundleInvalid: If it is not a varagity manifest at all.
    """
    try:
        document: Any = json.loads(raw)
    except ValueError as error:  # malformed JSON, or not UTF-8
        raise BundleInvalid(
            f"Its {MANIFEST_NAME} is not a Varagity manifest (not JSON)."
        ) from error
    if isinstance(document, dict) and document.get("format") == BUNDLE_FORMAT:
        version = document.get("version")
        if type(version) is int and version != BUNDLE_VERSION:
            raise BundleIncompatible(
                f"Bundle format version {version} — this stack reads version {BUNDLE_VERSION}."
            )
    try:
        return BundleManifest.model_validate(document)
    except ValidationError as error:
        raise BundleInvalid(
            f"Its {MANIFEST_NAME} is not a Varagity manifest ({error.error_count()} problem(s))."
        ) from error


def read_manifest(path: Path) -> BundleManifest:
    """Read a bundle's manifest — and only its manifest.

    The zip's central directory locates ``bundle.json``; no other member is
    opened, so listing a folder of large bundles stays cheap.

    Args:
        path: The bundle file.

    Returns:
        The validated manifest.

    Raises:
        BundleInvalid: If the file is not a readable zip, has no
            ``bundle.json``, or its manifest is not a varagity manifest.
        BundleIncompatible: If the manifest is of another format version.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            raw = _manifest_bytes(archive)
    except BundleError as error:
        raise type(error)(f"{path.name}: {error}") from error
    except _UNREADABLE_ZIP as error:
        raise BundleInvalid(f"{path.name}: {_unreadable(error)}") from error
    try:
        return _parse_manifest(raw)
    except BundleError as error:
        raise type(error)(f"{path.name}: {error}") from error


def _unsafe(info: zipfile.ZipInfo) -> str | None:
    """Say why a zip entry is unsafe to treat as a bundle member, if it is.

    Nothing here is extracted by name yet, but a restore must never act on a
    member that could land outside its target: the rule set is the classic
    zip-slip list, applied to every entry, listed in the manifest or not.

    Args:
        info: One entry of the zip's central directory.

    Returns:
        The problem in a few words, or ``None`` for a safe entry.
    """
    name = info.filename
    if "\\" in name:
        return "a backslash in its path"
    if name.startswith("/") or _DRIVE_PREFIX.match(name):
        return "an absolute path"
    parts = (name[:-1] if name.endswith("/") else name).split("/")
    if ".." in parts:
        return "it climbs out of the bundle"
    if any(part in ("", ".") for part in parts):
        return "an empty or '.' path segment"
    if stat.S_ISLNK(info.external_attr >> 16):
        return "a symlink"
    return None


def _check_members(infos: list[zipfile.ZipInfo], manifest: BundleManifest, corpus: Corpus) -> None:
    """Hold the zip's entries to the manifest and to the corpus's allowlist.

    Args:
        infos: The zip's central directory.
        manifest: The bundle's (already validated) manifest.
        corpus: The corpus being restored.

    Raises:
        BundleInvalid: For an unsafe, duplicated, unexpected, missing or
            unrecorded member, or one whose recorded size differs from the
            zip's (a damaged or hand-edited bundle).
        BundleIncompatible: If this stack can't restore the corpus at all.
    """
    allowed = _CORPUS_MEMBERS.get(corpus)
    if allowed is None:
        raise BundleIncompatible(
            f"This stack cannot restore {_CORPUS_NOUNS.get(corpus, corpus)} backups yet."
        )
    for info in infos:
        problem = _unsafe(info)
        if problem is not None:
            raise BundleInvalid(f"Unsafe member {info.filename!r}: {problem}.")
    for names, holder in (
        ([info.filename for info in infos], "The zip holds"),
        ([member.name for member in manifest.members], f"{MANIFEST_NAME} lists"),
    ):
        repeated = sorted(name for name, count in Counter(names).items() if count > 1)
        if repeated:
            raise BundleInvalid(f"{holder} {repeated[0]!r} more than once.")
    entries = {info.filename: info for info in infos if info.filename != MANIFEST_NAME}
    recorded = {member.name for member in manifest.members}
    unexpected = sorted((set(entries) | recorded) - allowed)
    if unexpected:
        raise BundleInvalid(f"Unexpected member(s): {', '.join(unexpected)}.")
    missing = sorted(recorded - set(entries))
    if missing:
        raise BundleInvalid(f"Missing member(s): {', '.join(missing)}.")
    unrecorded = sorted(allowed - recorded)
    if unrecorded:
        raise BundleInvalid(f"{MANIFEST_NAME} does not record {', '.join(unrecorded)}.")
    for member in manifest.members:
        actual = entries[member.name].file_size
        if actual != member.size_bytes:
            raise BundleInvalid(
                f"{member.name} is {actual:,} bytes in the zip but {member.size_bytes:,} in "
                f"{MANIFEST_NAME} — the bundle is damaged."
            )
    if corpus == "rag":
        _check_rag_layout(manifest, entries["embeddings.f32"].file_size)


def _check_rag_layout(manifest: BundleManifest, matrix_bytes: int) -> None:
    """Hold a RAG bundle's vector matrix to the row count its manifest records.

    ``embeddings.f32`` is ``chunks × embedding_dim`` float32s; any other
    size means its rows can't line up with ``chunks.jsonl``'s lines, which
    a restore would only discover after loading part of it.

    Args:
        manifest: The bundle's manifest.
        matrix_bytes: ``embeddings.f32``'s size in the zip.

    Raises:
        BundleInvalid: If the counts are missing or the matrix has the wrong
            size.
    """
    chunks = manifest.counts.get("chunks")
    documents = manifest.counts.get("documents")
    if chunks is None or documents is None or chunks < 0 or documents < 0:
        raise BundleInvalid(f"{MANIFEST_NAME} does not record its document and chunk counts.")
    expected = chunks * manifest.embedding_dim * 4
    if matrix_bytes != expected:
        raise BundleInvalid(
            f"embeddings.f32 is {matrix_bytes:,} bytes, but {chunks:,} chunk(s) of "
            f"{manifest.embedding_dim} float32s take {expected:,} — the bundle is damaged."
        )


def _check_compatible(manifest: BundleManifest) -> None:
    """Refuse a bundle from another embedding space than this stack's.

    Args:
        manifest: The bundle's manifest.

    Raises:
        BundleIncompatible: On an embedding model or dimension mismatch.
    """
    settings = get_settings()
    if manifest.embedding_model != settings.EMBEDDING_MODEL:
        raise BundleIncompatible(
            f"Embedded with {manifest.embedding_model} — this stack embeds with "
            f"{settings.EMBEDDING_MODEL}."
        )
    if manifest.embedding_dim != settings.EMBEDDING_DIM:
        raise BundleIncompatible(
            f"{manifest.embedding_dim}-dimensional embeddings — this stack embeds into "
            f"{settings.EMBEDDING_DIM} dimensions."
        )


def _inspect_archive(archive: zipfile.ZipFile, corpus: Corpus) -> BundleVerdict:
    """Judge an open bundle from its central directory and manifest alone.

    The checks run in the order that gives the most useful reason: how to
    read the file (format version), what it is (corpus), whether it is
    sound (members, sizes), and only then whether *this stack* can take it
    (embedding model and dimension).

    Args:
        archive: The open bundle.
        corpus: The corpus being restored.

    Returns:
        The verdict.
    """
    try:
        manifest = _parse_manifest(_manifest_bytes(archive))
    except BundleError as error:
        return BundleVerdict(manifest=None, problem=error)
    try:
        if manifest.corpus != corpus:
            raise BundleIncompatible(
                f"A {_CORPUS_NOUNS[manifest.corpus]} backup — it cannot restore the "
                f"{_CORPUS_NOUNS.get(corpus, corpus)}."
            )
        _check_members(archive.infolist(), manifest, corpus)
        _check_compatible(manifest)
    except BundleError as error:
        return BundleVerdict(manifest=manifest, problem=error)
    return BundleVerdict(manifest=manifest)


def inspect_bundle(path: Path, corpus: Corpus) -> BundleVerdict:
    """The cheap verdict the backups list shows for one bundle.

    Reads the zip's central directory and ``bundle.json`` — never a data
    member — and checks everything those can tell: the format version, the
    corpus, the member allowlist and zip-slip rules, each member's recorded
    size against the zip's, and the embedding model and dimension against
    this stack's settings. What it can't see is a member damaged *inside*
    an intact zip with unchanged sizes; :func:`verify_bundle` hashes for
    that.

    Args:
        path: The bundle file.
        corpus: The corpus it would restore.

    Returns:
        The verdict (it never raises for a bad file — the problem rides
        the verdict).
    """
    try:
        with zipfile.ZipFile(path) as archive:
            return _inspect_archive(archive, corpus)
    except _UNREADABLE_ZIP as error:
        return BundleVerdict(manifest=None, problem=_unreadable(error))


class MemberStream:
    """The read end of one bundle member, hashed and counted as it is read.

    Every byte handed to the reader passes through :meth:`read` or
    :meth:`readline`, so once the member is drained the digest covers it
    whole; :meth:`verify` then holds it to the manifest's record.

    Attributes:
        name: The member's name.
    """

    def __init__(self, raw: IO[bytes], record: BundleMember) -> None:
        """Wrap a member's decompressing read stream.

        Args:
            raw: The stream ``ZipFile.open(name)`` returned.
            record: What the manifest recorded for the member.
        """
        self.name = record.name
        self._raw = raw
        self._record = record
        self._hash = hashlib.sha256()
        self._size = 0

    def _failure(self, error: BaseException) -> BundleError:
        """Classify a read failure as damage or as an unreadable bundle.

        Args:
            error: What reading the member raised.

        Returns:
            ``BundleDamaged`` for a CRC or deflate failure (the bytes
            changed), ``BundleInvalid`` otherwise.
        """
        if isinstance(error, _DAMAGED_MEMBER):
            return BundleDamaged(
                f"{self.name} could not be read ({error}) — the bundle is damaged."
            )
        return BundleInvalid(f"{self.name} could not be read ({error}).")

    def _take(self, data: bytes) -> bytes:
        """Count and hash bytes on their way to the reader.

        Args:
            data: What the underlying stream returned.

        Returns:
            The same bytes.
        """
        self._hash.update(data)
        self._size += len(data)
        return data

    def read(self, size: int = -1) -> bytes:
        """Read up to ``size`` bytes (all that is left when negative).

        Args:
            size: The most bytes to return.

        Returns:
            The bytes; ``b""`` at the end of the member.

        Raises:
            BundleDamaged: If the member's bytes fail the zip's own checks.
            BundleInvalid: If the member can't be read for another reason.
        """
        try:
            data = self._raw.read(size)
        except _UNREADABLE_ZIP as error:
            raise self._failure(error) from error
        return self._take(data)

    def readline(self) -> bytes:
        """Read one line, newline included.

        Returns:
            The line; ``b""`` at the end of the member.

        Raises:
            BundleDamaged: If the member's bytes fail the zip's own checks.
            BundleInvalid: If the member can't be read for another reason.
        """
        try:
            data = self._raw.readline()
        except _UNREADABLE_ZIP as error:
            raise self._failure(error) from error
        return self._take(data)

    def __iter__(self) -> Iterator[bytes]:
        """Iterate the member's lines (a JSONL member's records).

        Yields:
            Each line, newline included.
        """
        while line := self.readline():
            yield line

    def verify(self) -> None:
        """Drain what the reader left, then hold the member to its record.

        Raises:
            BundleDamaged: If its size or sha256 differs from what the
                manifest recorded.
        """
        while self.read(_READ_BYTES):
            pass
        if self._size != self._record.size_bytes or self._hash.hexdigest() != self._record.sha256:
            raise BundleDamaged(
                f"{self.name} does not match its recorded sha256 — the bundle is damaged."
            )


class BundleSource:
    """An open bundle whose members are read through verifying streams.

    Attributes:
        manifest: The manifest the members are checked against.
    """

    def __init__(self, archive: zipfile.ZipFile, manifest: BundleManifest) -> None:
        """Bind the open zip to the manifest it is held to.

        Args:
            archive: The open bundle.
            manifest: Its (already inspected) manifest.
        """
        self._archive = archive
        self._records = {member.name: member for member in manifest.members}
        self.manifest = manifest

    @contextmanager
    def member(self, name: str) -> Iterator[MemberStream]:
        """Open one data member for reading, verified when the block exits.

        Several members may be open at once (zipfile reads each through its
        own position in the shared file), which is how a restore reads the
        chunk lines and their vectors side by side.

        Args:
            name: The member's name.

        Yields:
            The member's verifying stream. When the ``with`` block exits
            cleanly, whatever the reader left is drained and the member's
            size and sha256 are checked against the manifest.

        Raises:
            BundleInvalid: If the manifest records no such member, or the zip
                can't open it.
            BundleDamaged: If the member's bytes don't match its record.
        """
        record = self._records.get(name)
        if record is None:
            raise BundleInvalid(f"Missing member(s): {name}.")
        try:
            raw = self._archive.open(name)
        except KeyError:
            raise BundleInvalid(f"Missing member(s): {name}.") from None
        except _UNREADABLE_ZIP as error:
            raise BundleInvalid(f"{name} could not be opened ({error}).") from error
        with raw:
            stream = MemberStream(raw, record)
            yield stream
            stream.verify()


@contextmanager
def open_bundle(path: Path, manifest: BundleManifest) -> Iterator[BundleSource]:
    """Open a bundle to read its members against its manifest.

    Args:
        path: The bundle file.
        manifest: Its manifest (from :func:`inspect_bundle` or
            :func:`verify_bundle`).

    Yields:
        The open bundle.

    Raises:
        BundleInvalid: If the file can't be opened as a zip.
    """
    try:
        archive = zipfile.ZipFile(path)
    except _UNREADABLE_ZIP as error:
        raise _unreadable(error) from error
    with archive:
        yield BundleSource(archive, manifest)


def verify_bundle(path: Path, corpus: Corpus) -> BundleManifest:
    """Run every check a restore needs before it touches a store.

    :func:`inspect_bundle`'s verdict, then the sha256 of every member —
    every byte of the bundle is read once. A restore refuses the bundle on
    any failure, so a damaged bundle can never restore.

    Args:
        path: The bundle file.
        corpus: The corpus it would restore.

    Returns:
        The bundle's manifest, now fully verified.

    Raises:
        BundleIncompatible: If this stack must not restore the bundle.
        BundleDamaged: If a member's bytes don't match its recorded sha256.
        BundleInvalid: If the bundle is otherwise unsound.
    """
    manifest = inspect_bundle(path, corpus).manifest_or_raise()
    with open_bundle(path, manifest) as source:
        for member in manifest.members:
            # Opening and closing a member is the whole check: the clean
            # exit drains it through the hash and compares the digest.
            with source.member(member.name):
                pass
    return manifest


def list_bundles(directory: Path, corpus: Corpus) -> list[BundleListing]:
    """List the bundles in an exports folder, newest first, each with its verdict.

    Hidden files (in-flight partials included) and non-``.zip`` files
    are skipped, and so is anything that resolves outside the folder (a
    symlink pointing elsewhere is not a bundle this folder holds). A visible
    zip that isn't a readable bundle is still listed — with no manifest and
    its reason — because the user put it there and the panel should say what
    it is.

    Args:
        directory: The exports folder (a missing folder lists nothing).
        corpus: The corpus the folder belongs to (what each bundle is judged
            against).

    Returns:
        One listing per bundle file, newest mtime first.
    """
    if not directory.is_dir():
        return []
    root = directory.resolve()
    listings: list[BundleListing] = []
    for entry in directory.iterdir():
        if not is_bundle_name(entry.name):
            continue
        resolved = resolve_contained(entry, root)
        if resolved is None or not resolved.is_file():
            continue
        try:
            info = resolved.stat()
        except OSError:
            continue  # vanished between the listing and the stat
        verdict = inspect_bundle(resolved, corpus)
        if verdict.problem is not None:
            logger.debug("listing %s as not restorable: %s", entry.name, verdict.reason)
        listings.append(
            BundleListing(
                name=entry.name,
                path=resolved,
                size_bytes=info.st_size,
                modified_at=datetime.fromtimestamp(info.st_mtime, tz=UTC),
                mtime_ns=info.st_mtime_ns,
                verdict=verdict,
            )
        )
    listings.sort(key=lambda listing: (listing.modified_at, listing.name), reverse=True)
    return listings
