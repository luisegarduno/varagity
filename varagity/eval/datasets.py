"""Golden Q&A dataset: loading, validation, and identity resolution (spec §16).

The golden set (``data/eval/golden_qa.jsonl``) is hand-authored over
``tests/fixtures/corpus``. Each line is one query with the chunks required
to answer it, identified **portably** by relative source path +
``chunk_index`` — never by ``chunk_id`` directly, whose ``doc_id`` half
depends on file bytes:

    {"query": "…", "relevant": [{"rel_source": "aurora_station.md", "chunk_index": 1}]}

``rel_source`` is the document's POSIX path relative to the eval corpus
root — exactly the string the ingest loader hashes into ``doc_id`` (plan
decision #6) — so an entry resolves to concrete ``chunk_id``s from the
corpus files alone: ``derive_doc_id(rel_source, content_hash(bytes))``
plus the deterministic ``{doc_id}::{chunk_index}`` composition. No store
round-trip is needed, and the resolution is identical on every machine.

``chunk_index`` values assume the chunk boundaries produced by the pinned
eval pipeline settings (``recursive_character``, size 400 / overlap 50 —
see ``varagity.eval.evaluate.PINNED_EVAL_SETTINGS``); re-author the golden
set if those pins change.
"""

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from varagity.stores.records import content_hash, derive_doc_id

logger = logging.getLogger(__name__)


class GoldenChunkRef(BaseModel):
    """Portable identity of one relevant chunk.

    ``chunk_index`` anchors the ref to the **pinned** chunk boundaries (the
    matrix's fixed configuration); ``fact`` anchors it to **content**, which
    survives any boundary change — the chunker sweep re-resolves refs by
    scanning the strategy-true chunks for the fact (spec_v2 §7.4), since an
    index authored against one strategy's boundaries points at arbitrary
    text under another's.

    Attributes:
        rel_source: Document path relative to the eval corpus root
            (POSIX-style, e.g. ``"aurora_station.md"`` — the same string
            the loader derives ``doc_id`` from).
        chunk_index: The chunk's position within its document (0-based),
            under the pinned eval chunk boundaries.
        fact: A short verbatim snippet of the chunk's source text that
            uniquely identifies the golden content within its document
            (matched case-insensitively — OCR extraction may case-shift).
            Optional for the fixed matrix; required for a ref to
            participate in the chunker sweep.
    """

    rel_source: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    fact: str | None = Field(default=None, min_length=1)


class GoldenEntry(BaseModel):
    """One golden query and every chunk required to answer it.

    Attributes:
        query: The evaluation question, as a user would ask it.
        relevant: The chunks a perfect retriever would return (≥ 1).
    """

    query: str = Field(min_length=1)
    relevant: list[GoldenChunkRef] = Field(min_length=1)


class ResolvedGoldenEntry(BaseModel):
    """A golden entry with its refs resolved to concrete ``chunk_id``s.

    Attributes:
        query: The evaluation question.
        relevant: The original portable refs (kept for reporting).
        chunk_ids: One resolved ``{doc_id}::{chunk_index}`` per ref, in
            ref order.
    """

    query: str
    relevant: list[GoldenChunkRef]
    chunk_ids: list[str]


def load_golden(path: Path) -> list[GoldenEntry]:
    """Load and validate the golden dataset from a JSONL file.

    Blank lines are ignored; any malformed line fails the load loudly —
    a silently skipped golden query would quietly inflate every score.

    Args:
        path: The ``golden_qa.jsonl`` file.

    Returns:
        The validated entries, in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a line is not valid JSON or fails schema validation
            (message includes the 1-based line number).
    """
    entries: list[GoldenEntry] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entries.append(GoldenEntry.model_validate(json.loads(line)))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: not valid JSON — {exc}") from exc
        except ValidationError as exc:
            raise ValueError(f"{path}:{line_no}: invalid golden entry — {exc}") from exc
    if not entries:
        raise ValueError(f"{path}: golden dataset is empty")
    logger.info("loaded %d golden queries from %s", len(entries), path)
    return entries


def resolve_golden(entries: list[GoldenEntry], corpus_root: Path) -> list[ResolvedGoldenEntry]:
    """Resolve portable refs to the ``chunk_id``s the stores will hold.

    Reproduces the loader's identity derivation (plan decision #6) from
    the corpus files themselves: ``doc_id`` hashes ``rel_source`` together
    with the file's byte hash, so resolution needs no ingested store and
    is stable across machines and OCR engines (bytes, not extracted text).

    Whether a resolved ``chunk_id`` actually exists (i.e. ``chunk_index``
    is within the document's real chunk count) is checked by the harness
    after ingest — it depends on the live chunking, not on this file.

    Args:
        entries: The validated golden entries.
        corpus_root: The eval corpus root the refs are relative to.

    Returns:
        One resolved entry per input entry, in order.

    Raises:
        FileNotFoundError: If a ref names a file missing from the corpus.
    """
    doc_ids: dict[str, str] = {}
    resolved: list[ResolvedGoldenEntry] = []
    for entry in entries:
        chunk_ids: list[str] = []
        for ref in entry.relevant:
            if ref.rel_source not in doc_ids:
                source = corpus_root / ref.rel_source
                if not source.is_file():
                    raise FileNotFoundError(
                        f"golden ref {ref.rel_source!r} not found under {corpus_root} — "
                        "the golden set and the eval corpus are out of sync"
                    )
                doc_ids[ref.rel_source] = derive_doc_id(
                    ref.rel_source, content_hash(source.read_bytes())
                )
            chunk_ids.append(f"{doc_ids[ref.rel_source]}::{ref.chunk_index}")
        resolved.append(
            ResolvedGoldenEntry(query=entry.query, relevant=entry.relevant, chunk_ids=chunk_ids)
        )
    return resolved
