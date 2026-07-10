"""Corpus discovery: scan the docs directory and bucket files by parser.

Bucketing (spec §9.1): ``.txt`` and ``.md`` share one extraction path
(``text_like``); ``.pdf`` needs Docling (``pdf``). Extensions outside the
``ALLOWED_EXTENSIONS`` whitelist are ignored (logged at DEBUG).
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from varagity.config import get_settings
from varagity.debug.show import check_verbose, v_discover

logger = logging.getLogger(__name__)

# Extension → bucket routing (spec §9.1 table). The whitelist decides *if* a
# file is ingested; this decides *which* parser family handles it.
_TEXT_LIKE_EXTENSIONS = frozenset({".txt", ".md"})
_PDF_EXTENSIONS = frozenset({".pdf"})


@dataclass
class Buckets:
    """Discovered corpus files, grouped by extraction path.

    Attributes:
        text_like: ``.txt`` / ``.md`` files (parsed by ``parsers/text.py``).
        pdf: ``.pdf`` files (parsed by ``parsers/pdf.py`` from Phase 7).
    """

    text_like: list[Path] = field(default_factory=list)
    pdf: list[Path] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Total number of bucketed files.

        Returns:
            The combined size of all buckets.
        """
        return len(self.text_like) + len(self.pdf)


def discover_documents(docs_path: str, verbose: int | None = None) -> Buckets:
    """Recursively scan ``docs_path`` and bucket ingestable files.

    A missing directory is not an error: the app re-scans on every start
    (spec §9.1), so an empty/unmounted corpus logs a warning and yields empty
    buckets rather than crashing.

    Args:
        docs_path: Directory to scan (usually ``settings.DOCS_PATH``).
        verbose: Console verbosity (0–2); defaults to
            ``settings.DEFAULT_VERBOSE``.

    Returns:
        The bucketed file paths, each bucket sorted for determinism.

    Raises:
        ValueError: If ``verbose`` is invalid.
    """
    settings = get_settings()
    verbose = check_verbose(settings.DEFAULT_VERBOSE if verbose is None else verbose)
    buckets = Buckets()

    root = Path(docs_path)
    if not root.is_dir():
        logger.warning("docs directory %s does not exist — nothing to ingest", docs_path)
        v_discover(buckets, verbose=verbose)
        return buckets

    allowed = settings.allowed_extension_set
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        extension = path.suffix.lower()
        if extension not in allowed:
            logger.debug("ignoring %s (extension %r not in ALLOWED_EXTENSIONS)", path, extension)
            continue
        if extension in _TEXT_LIKE_EXTENSIONS:
            buckets.text_like.append(path)
        elif extension in _PDF_EXTENSIONS:
            buckets.pdf.append(path)
        else:
            logger.warning(
                "%s is allowed by ALLOWED_EXTENSIONS but has no ingestion bucket; skipping", path
            )

    v_discover(buckets, verbose=verbose)
    return buckets
