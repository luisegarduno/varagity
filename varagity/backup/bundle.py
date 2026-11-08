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

This module knows the container, not the corpora: what goes *inside* a bundle
is :mod:`varagity.backup.rag`'s business (and, later, the graph's), and
nothing here knows about HTTP.
"""

import hashlib
import logging
import os
import stat
import zipfile
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Literal

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
        manifest: Its manifest, or ``None`` when the file isn't a readable
            varagity bundle (copied in by hand, damaged, …).
    """

    name: str
    path: Path
    size_bytes: int
    modified_at: datetime
    manifest: BundleManifest | None


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
    """
    try:
        with zipfile.ZipFile(path) as archive:
            try:
                info = archive.getinfo(MANIFEST_NAME)
            except KeyError:
                raise BundleInvalid(f"{path.name} has no {MANIFEST_NAME}") from None
            if info.file_size > _MAX_MANIFEST_BYTES:
                raise BundleInvalid(
                    f"{path.name}'s {MANIFEST_NAME} is {info.file_size} bytes — not a manifest"
                )
            raw = archive.read(info)
    except BundleInvalid:
        raise
    except _UNREADABLE_ZIP as error:
        raise BundleInvalid(f"{path.name} is not a readable zip ({error})") from error
    try:
        return BundleManifest.model_validate_json(raw)
    except ValidationError as error:
        raise BundleInvalid(
            f"{path.name}'s {MANIFEST_NAME} is not a varagity manifest "
            f"({error.error_count()} problem(s))"
        ) from error


def list_bundles(directory: Path) -> list[BundleListing]:
    """List the bundles in an exports folder, newest first.

    Hidden files (in-flight partials included) and non-``.zip`` files
    are skipped, and so is anything that resolves outside the folder (a
    symlink pointing elsewhere is not a bundle this folder holds). A visible
    zip that isn't a readable bundle is still listed — with no manifest —
    because the user put it there and the panel should say what it is.

    Args:
        directory: The exports folder (a missing folder lists nothing).

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
        try:
            manifest: BundleManifest | None = read_manifest(resolved)
        except BundleInvalid as error:
            logger.debug("listing %s without a manifest: %s", entry.name, error)
            manifest = None
        listings.append(
            BundleListing(
                name=entry.name,
                path=resolved,
                size_bytes=info.st_size,
                modified_at=datetime.fromtimestamp(info.st_mtime, tz=UTC),
                manifest=manifest,
            )
        )
    listings.sort(key=lambda listing: (listing.modified_at, listing.name), reverse=True)
    return listings
