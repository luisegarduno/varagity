"""PPTX→PDF conversion for previews (ADR-010).

Headless LibreOffice Impress renders a deck to a PDF whose page N *is*
slide N — the same identity docling relies on — so :mod:`~varagity.preview.locate`
and :mod:`~varagity.preview.render` work unchanged on the converted
artifact. Conversions are cached by ``doc_id`` (content-addressed, so a
hit can never be stale), serialized behind a module lock (LibreOffice's
profile is single-instance), and bounded by ``PREVIEW_CONVERT_TIMEOUT_S``.
A host-mode run without LibreOffice degrades (``conversion_unavailable``),
never crashes.
"""

import logging
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# LibreOffice serializes on its user profile: even with per-call isolated
# profiles, concurrent soffice processes fight over CPU for no benefit on
# a single-user box, and the lock lets losers of the race return the
# winner's cached artifact instead of re-converting.
_CONVERT_LOCK = threading.Lock()


class ConversionUnavailable(Exception):
    """No converter is available to this process (degrade, never crash).

    Raised when LibreOffice is absent — a host-mode run without it loses
    only PPTX previews (the route answers ``conversion_unavailable``).
    """


class ConversionFailed(Exception):
    """A conversion ran and failed (non-zero exit, timeout, missing output)."""


def conversion_cache_path(doc_id: str) -> Path:
    """Return the cached-PDF path for one document's conversion.

    Content-addressed: ``doc_id`` hashes the source path and its byte
    content, so a cache hit can never be stale. The cache lives in the
    process's temp directory — container-ephemeral by design (a restart
    re-pays one conversion per deck).

    Args:
        doc_id: The document's stable id.

    Returns:
        The cache path (existence not implied).
    """
    return Path(tempfile.gettempdir()) / "varagity-preview" / f"{doc_id}.pdf"


def ensure_pdf(source: Path, doc_id: str, *, timeout_s: int) -> Path:
    """Return a PDF rendition of ``source``, converting on first use.

    Cache first — a previously converted deck answers instantly (and keeps
    answering even if LibreOffice later disappears from the image). On a
    miss, one ``soffice --headless --convert-to pdf`` run executes under
    the module lock with a throwaway ``-env:UserInstallation`` profile,
    and its output moves atomically into the cache path (a concurrent
    reader never sees a partial file).

    Args:
        source: The source document (a ``.pptx``).
        doc_id: The document's stable id (the cache key).
        timeout_s: Conversion timeout in seconds (``PREVIEW_CONVERT_TIMEOUT_S``).

    Returns:
        The pdfium-openable converted PDF.

    Raises:
        ConversionUnavailable: LibreOffice (``soffice``) is not on ``PATH``.
        ConversionFailed: The conversion ran and failed — non-zero exit,
            timeout, or no PDF produced.
    """
    cached = conversion_cache_path(doc_id)
    if cached.is_file():
        return cached
    soffice = shutil.which("soffice")
    if soffice is None:
        raise ConversionUnavailable(
            "LibreOffice (soffice) is not on PATH — PPTX previews degrade to full text"
        )
    with _CONVERT_LOCK:
        if cached.is_file():  # a concurrent request converted it while we waited
            return cached
        cached.parent.mkdir(parents=True, exist_ok=True)
        # Scratch inside the cache directory: the final move is a same-
        # filesystem rename, so the cache path is atomically all-or-nothing
        # (partial-file pattern of the upload route).
        with tempfile.TemporaryDirectory(prefix=".convert-", dir=cached.parent) as scratch:
            produced = _run_soffice(soffice, source, Path(scratch), timeout_s=timeout_s)
            shutil.move(produced, cached)
    logger.info("converted %s to %s for preview", source.name, cached)
    return cached


def _run_soffice(soffice: str, source: Path, scratch: Path, *, timeout_s: int) -> Path:
    """Run one headless conversion into ``scratch`` (caller holds the lock).

    Args:
        soffice: Resolved path of the LibreOffice binary.
        source: The document to convert.
        scratch: Private working directory (profile + output live here).
        timeout_s: Subprocess timeout in seconds.

    Returns:
        The produced PDF, still inside ``scratch``.

    Raises:
        ConversionFailed: Non-zero exit, timeout, or no PDF produced.
    """
    out_dir = scratch / "out"
    out_dir.mkdir()
    command = [
        soffice,
        "--headless",
        "--norestore",
        # A per-call throwaway profile: no clash with any other LibreOffice
        # instance and no stale-profile lock files surviving a crash.
        f"-env:UserInstallation={(scratch / 'profile').as_uri()}",
        "--convert-to",
        "pdf",
        "--outdir",
        str(out_dir),
        str(source),
    ]
    try:
        completed = subprocess.run(command, timeout=timeout_s, capture_output=True, check=False)
    except subprocess.TimeoutExpired as error:
        raise ConversionFailed(f"soffice timed out after {timeout_s}s on {source.name}") from error
    if completed.returncode != 0:
        stderr = completed.stderr.decode(errors="replace").strip()
        raise ConversionFailed(f"soffice exited {completed.returncode} on {source.name}: {stderr}")
    produced = out_dir / f"{source.stem}.pdf"
    if not produced.is_file():
        # Exit 0 without output happens (e.g. unreadable input filters).
        stderr = completed.stderr.decode(errors="replace").strip()
        raise ConversionFailed(f"soffice produced no PDF for {source.name}: {stderr}")
    return produced
