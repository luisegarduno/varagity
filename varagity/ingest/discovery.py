"""Corpus discovery: scan the docs directory and bucket files by parser.

Bucketing (spec §9.1, spec_v2 §8.1): ``.txt`` and ``.md`` share one
extraction path (``text_like``); ``.pdf`` needs Docling with the OCR
fallback (``pdf``); ``.docx``/``.pptx``/``.xlsx`` share the no-OCR Docling
office path (``office``); ``.html``/``.htm`` likewise (``web``). Extensions
outside the ``ALLOWED_EXTENSIONS`` whitelist are ignored (logged at DEBUG).
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from varagity.config import get_settings
from varagity.debug.show import check_verbose, v_discover

logger = logging.getLogger(__name__)

# Extension → bucket routing (spec §9.1 / spec_v2 §8.1 tables). The whitelist
# decides *if* a file is ingested; this decides *which* parser family
# handles it.
_TEXT_LIKE_EXTENSIONS = frozenset({".txt", ".md"})
_PDF_EXTENSIONS = frozenset({".pdf"})
_OFFICE_EXTENSIONS = frozenset({".docx", ".pptx", ".xlsx"})
_WEB_EXTENSIONS = frozenset({".html", ".htm"})


@dataclass
class Buckets:
    """Discovered corpus files, grouped by extraction path.

    Attributes:
        text_like: ``.txt`` / ``.md`` files (parsed by ``parsers/text.py``).
        pdf: ``.pdf`` files (parsed by ``parsers/pdf.py``).
        office: ``.docx`` / ``.pptx`` / ``.xlsx`` files (parsed by
            ``parsers/office.py``).
        web: ``.html`` / ``.htm`` files (parsed by ``parsers/web.py``).
    """

    text_like: list[Path] = field(default_factory=list)
    pdf: list[Path] = field(default_factory=list)
    office: list[Path] = field(default_factory=list)
    web: list[Path] = field(default_factory=list)

    def by_bucket(self) -> tuple[tuple[str, list[Path]], ...]:
        """List every bucket with its name, in a stable order.

        The single enumeration point renderers iterate, so a future bucket
        added here appears everywhere without caller edits.

        Returns:
            ``(bucket_name, paths)`` pairs, one per bucket.
        """
        return (
            ("text_like", self.text_like),
            ("pdf", self.pdf),
            ("office", self.office),
            ("web", self.web),
        )

    @property
    def total(self) -> int:
        """Total number of bucketed files.

        Returns:
            The combined size of all buckets.
        """
        return sum(len(paths) for _name, paths in self.by_bucket())


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
        elif extension in _OFFICE_EXTENSIONS:
            buckets.office.append(path)
        elif extension in _WEB_EXTENSIONS:
            buckets.web.append(path)
        else:
            logger.warning(
                "%s is allowed by ALLOWED_EXTENSIONS but has no ingestion bucket; skipping", path
            )

    v_discover(buckets, verbose=verbose)
    return buckets
