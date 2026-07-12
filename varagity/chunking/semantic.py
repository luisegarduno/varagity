"""Embedding-similarity boundary detection (spec_v2 §7, plan decision #12).

Groups adjacent sentences and splits **where the topic shifts**: each
segment is embedded (with one neighbor of context on each side, smoothing
single-sentence noise) through the existing infinity embeddings client in
**passage mode**, consecutive-segment cosine distances are computed, and a
boundary is drawn wherever the distance is a statistical outlier — above
the 95th percentile of the document's distances (LangChain's default
``percentile`` breakpoint; the ``gradient`` variant is avoided per its open
KeyError bug — plan decision #12).

Two guarantees bound the output:

* **Exact reconstruction below the ceiling** — segments are sliced from the
  original text with their separators attached, so a chunk is a verbatim
  substring and un-resplit chunks concatenate back to the document.
* **The 512-token ceiling** — similarity groups are meaning-sized, not
  budget-sized, so any group exceeding ``CHUNK_SIZE`` (**tokens**, like
  ``token_based``) is re-split with the token-denominated recursive
  splitter (``CHUNK_OVERLAP`` tokens of overlap applies only there).

This strategy embeds at **ingest time** — one passage request per segment
window, batched by ``EMBEDDING_BATCH_SIZE`` — an added ingest cost the
Phase 6 sweep measures before any default changes (ADR-008). The client
resolves lazily via ``get_model("embedding")`` per split, so the registry's
import-time instantiation performs no I/O; tests inject a deterministic
fake instead.
"""

import math
import re
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from varagity.chunking.base import register, warn_near_token_ceiling
from varagity.config import get_settings
from varagity.debug.show import check_verbose, v_chunk
from varagity.models.embeddings import EmbeddingsClient
from varagity.models.registry import get_model
from varagity.tokens import count_tokens

# Segment boundary: after sentence-ending punctuation + whitespace, or any
# newline run (markdown headings/list items rarely end with punctuation).
_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+|\n+")

# Neighboring segments embedded together with each segment (smoothing).
_BUFFER_SIZE = 1

# Consecutive-segment cosine distances above this percentile become chunk
# boundaries (LangChain's `percentile` default).
_BREAKPOINT_PERCENTILE = 95.0


def _segments(text: str) -> list[str]:
    """Slice text into sentence-ish segments that reconstruct exactly.

    Boundaries fall after sentence enders and newline runs; each separator
    stays attached to the segment it ends, so ``"".join(segments) == text``.
    Whitespace-only slices are merged into their predecessor (they carry no
    meaning to embed).

    Args:
        text: The document text.

    Returns:
        The segments, in order (empty list for empty text).
    """
    spans: list[str] = []
    start = 0
    for match in _BOUNDARY_RE.finditer(text):
        spans.append(text[start : match.end()])
        start = match.end()
    if start < len(text):
        spans.append(text[start:])
    segments: list[str] = []
    for span in spans:
        if segments and not span.strip():
            segments[-1] += span
        else:
            segments.append(span)
    return segments


def _cosine_distance(left: list[float], right: list[float]) -> float:
    """Cosine distance ``1 - cos(left, right)``, tolerating zero vectors.

    Args:
        left: One embedding.
        right: The other embedding.

    Returns:
        The distance (``1.0`` when either vector has zero norm).
    """
    norms = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(y * y for y in right))
    if norms == 0.0:
        return 1.0
    return 1.0 - sum(x * y for x, y in zip(left, right, strict=True)) / norms


def _percentile(values: list[float], percentile: float) -> float:
    """Linear-interpolated percentile (numpy's default method, dependency-free).

    Args:
        values: The sample (non-empty).
        percentile: The percentile in ``[0, 100]``.

    Returns:
        The interpolated percentile value.

    Raises:
        ValueError: If ``values`` is empty.
    """
    if not values:
        raise ValueError("percentile of an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


@register("semantic")
class SemanticStrategy:
    """Splits where consecutive-segment embedding similarity drops."""

    def __init__(self, *, embeddings: EmbeddingsClient | None = None) -> None:
        """Create the strategy.

        Args:
            embeddings: Embeddings client for boundary detection; resolved
                via the model registry (``get_model("embedding")``) per call
                when omitted. Tests inject a deterministic fake.
        """
        self._embeddings = embeddings

    def split(
        self, text: str, *, source_meta: dict[str, Any], verbose: int | None = None
    ) -> list[Document]:
        """Split a document's text at embedding-similarity boundaries.

        Args:
            text: The full document text.
            source_meta: Provenance copied into every chunk's metadata.
            verbose: Console verbosity (0–2); defaults to
                ``settings.DEFAULT_VERBOSE``.

        Returns:
            The chunks in document order, each carrying ``source_meta`` plus
            its ``chunk_index``.

        Raises:
            ValueError: If ``verbose`` is invalid.
            openai.APIError: If segment embedding still fails after the
                client's retries.
        """
        settings = get_settings()
        verbose = check_verbose(settings.DEFAULT_VERBOSE if verbose is None else verbose)
        groups = self._similarity_groups(_segments(text))
        resplitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=settings.CHUNK_SIZE,  # tokens, like token_based
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        chunks: list[Document] = []
        for group in groups:
            group_text = group.strip()
            if not group_text:
                continue
            texts = (
                resplitter.split_text(group_text)
                if count_tokens(group_text) > settings.CHUNK_SIZE
                else [group_text]
            )
            chunks.extend(
                Document(page_content=piece, metadata=dict(source_meta)) for piece in texts
            )
        for chunk_index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = chunk_index
        warn_near_token_ceiling(chunks, strategy="semantic")
        v_chunk(chunks, verbose=verbose)
        return chunks

    def _similarity_groups(self, segments: list[str]) -> list[str]:
        """Group segments between similarity-outlier boundaries.

        Fewer than three segments yield a single group without an
        embeddings call (one consecutive distance can never be its own
        outlier under a strict-greater percentile cut).

        Args:
            segments: The document's segments, in order.

        Returns:
            The grouped texts, in order (verbatim concatenations of their
            segments).
        """
        if not segments:
            return []
        if len(segments) < 3:
            return ["".join(segments)]
        client = self._embeddings if self._embeddings is not None else get_model("embedding")
        windows = [
            "".join(segments[max(0, i - _BUFFER_SIZE) : i + _BUFFER_SIZE + 1]).strip()
            for i in range(len(segments))
        ]
        vectors = client.embed_passages(windows, verbose=0)
        distances = [
            _cosine_distance(left, right) for left, right in zip(vectors, vectors[1:], strict=False)
        ]
        threshold = _percentile(distances, _BREAKPOINT_PERCENTILE)
        groups: list[str] = []
        current: list[str] = []
        for index, segment in enumerate(segments):
            current.append(segment)
            if index < len(distances) and distances[index] > threshold:
                groups.append("".join(current))
                current = []
        if current:
            groups.append("".join(current))
        return groups
