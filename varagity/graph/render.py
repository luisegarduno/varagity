"""Rendering messages into what the graph engine eats (spec_graphrag §5.2, §10.2).

Rendering lives here rather than in the adapter — pure functions over
:class:`~varagity.graph.sources.base.SourceMessage` are where most adapter
correctness actually is, and they are testable without the engine installed.
The shipped engine is document-shaped (ADR-017), so messages become thread
transcripts split on day boundaries (:func:`thread_transcripts`). The
episode-shaped rendering the bake-off's Graphiti seat ate went out with that
adapter (stage-2 loser removal); git history has it if an episode-shaped
engine ever re-enters the seam.

Upstream of it is :func:`merge_batches`, the upsert seam: source
files overlap (a re-export of a grown ``chat.db`` is a superset of the older
one), so messages are guid-deduped across batches before anything is
rendered. Rendering is then a **pure function of the merged messages**: the
same thread-days always produce the same :attr:`TranscriptDoc.doc_key` and
the same text, so an unchanged conversation stays upsert-identical no matter
how many times it is re-uploaded — which is exactly what
:mod:`varagity.graph.manifest`'s content hashes turn into a build diff.

Purity is why nothing here reads :func:`~varagity.config.get_settings`: the
owner's display label rides in on
:attr:`~varagity.graph.sources.base.SourceMessage.sender_name`, which the
parser already resolved from ``GRAPH_OWNER_ALIASES``. Re-deriving it at
render time would let a settings change silently rewrite documents that the
engines have already indexed under the same key.
"""

import logging
from collections.abc import Iterator, Mapping, Sequence
from datetime import date
from typing import Any

from pydantic import BaseModel

from varagity.graph.sources.base import MessageBatch, SourceMessage

logger = logging.getLogger(__name__)

# Transcript size target. Both document-shaped engines chunk what they are
# given before extraction, so this is about keeping *one* document inside the
# extraction call's budget rather than about the engines' own chunk size:
# ~8,000 characters is ~2,000 tokens, which leaves an extraction prompt, its
# gleaning pass, and the completion comfortable room inside a 16,384-token
# LLM_CONTEXT_TOKENS window (llama.cpp hard-500s at the window rather than
# stopping gracefully).
DEFAULT_TRANSCRIPT_MAX_CHARS = 8000

# Depth cap for the provenance walk over an engine-native payload — engine
# payloads are finite, but a cycle or a pathologically nested structure must
# not take the harness down with a RecursionError.
_MAX_PAYLOAD_DEPTH = 12


class TranscriptDoc(BaseModel):
    """One rendered thread transcript, as the document-shaped engines see it.

    Attributes:
        doc_key: Stable document identity, ``{thread_id}::{day-span}`` (one
            day, or ``first..last`` when a document packs several). It is
            deliberately **not** derived from a ``doc_id``: the same
            thread-days rendered from an overlapping upload or a re-export
            of a grown database must land on the same key, and a file's
            ``doc_id`` changes with every byte added to it.
        thread_id: The conversation this document came from.
        thread_name: Human-facing thread label (the header's subject).
        text: The rendered transcript — a ``Thread:`` header line followed by
            one ``[YYYY-MM-DD HH:MM] Sender: text`` line per message, with
            folded reactions indented beneath the message they land on.
        message_guids: Every message in the document, in transcript order —
            the map back from an engine's document-grain citation to
            per-message provenance.
    """

    doc_key: str
    thread_id: str
    thread_name: str
    text: str
    message_guids: list[str]


def merge_batches(batches: Sequence[MessageBatch]) -> list[SourceMessage]:
    """Guid-merge parsed source files into one ordered message stream.

    The upsert seam (plan amendment, owner-approved 2026-07-25): overlapping
    ``chat.db`` uploads and re-exports of a growing database share message
    guids, so the union — not the concatenation — is what any engine should
    index. First occurrence wins: batches are processed in the order given,
    and a later batch's copy of an already-seen guid is dropped rather than
    overwriting (a re-export's rows are the same rows; preferring the first
    keeps a build reproducible regardless of upload order).

    Args:
        batches: Parsed source files, in caller order.

    Returns:
        Every distinct message, sorted by ``(timestamp, guid)`` — the same
        stable order the parser emits, so equal input renders equally.
    """
    merged: dict[str, SourceMessage] = {}
    duplicates = 0
    for batch in batches:
        for message in batch.messages:
            if message.guid in merged:
                duplicates += 1
                continue
            merged[message.guid] = message
    if duplicates:
        logger.info(
            "merged %d batch(es): %d message(s), %d duplicate guid(s) dropped",
            len(batches),
            len(merged),
            duplicates,
        )
    return sorted(merged.values(), key=lambda message: (message.timestamp, message.guid))


def thread_transcripts(
    messages: Sequence[SourceMessage],
    *,
    max_chars: int = DEFAULT_TRANSCRIPT_MAX_CHARS,
) -> list[TranscriptDoc]:
    """Render messages into per-thread transcript documents split on day boundaries.

    Days are the atom: consecutive days of one thread are packed into a
    document until the next day would push it past ``max_chars``, and a day
    is never split in two. That keeps :attr:`TranscriptDoc.doc_key` meaningful
    and stable, and costs nothing in practice because both document-shaped
    engines chunk documents themselves before extraction. A single day whose
    messages exceed the cap therefore becomes one oversized document, logged
    rather than silently truncated.

    Args:
        messages: Merged messages (any order; they are re-sorted here).
        max_chars: Size target per document, in characters, counted over the
            message lines (the header adds a short constant on top).

    Returns:
        One document per thread-day-span, threads in first-appearance order
        and documents in chronological order within a thread.

    Raises:
        ValueError: If ``max_chars`` is not positive.
    """
    if max_chars <= 0:
        raise ValueError(f"max_chars must be positive; got {max_chars}")
    docs: list[TranscriptDoc] = []
    for thread_id, thread_messages in _by_thread(messages).items():
        packed: list[SourceMessage] = []
        size = 0
        for day, day_messages in _by_day(thread_messages).items():
            block = len(_render_lines(day_messages)) + 1  # +1 for the joining newline
            if packed and size + block > max_chars:
                docs.append(_transcript_doc(thread_id, packed))
                packed, size = [], 0
            if not packed and block > max_chars:
                logger.info(
                    "thread %s on %s renders to %d chars (max_chars=%d) — kept whole: a day is "
                    "the transcript atom and the engines chunk documents themselves",
                    thread_id,
                    day.isoformat(),
                    block,
                    max_chars,
                )
            packed.extend(day_messages)
            size += block
        if packed:
            docs.append(_transcript_doc(thread_id, packed))
    return docs


def doc_guid_index(docs: Sequence[TranscriptDoc]) -> dict[str, list[str]]:
    """Index transcript documents by key, for mapping citations back to messages.

    Args:
        docs: Rendered transcripts.

    Returns:
        ``doc_key`` → its message guids, in transcript order.
    """
    return {doc.doc_key: list(doc.message_guids) for doc in docs}


def guids_in_payload(payload: Any, index: Mapping[str, Sequence[str]]) -> list[str]:
    """Recover message provenance from an engine-native payload.

    Every adapter hands its engine document keys as the document ids/file
    names it indexes under, so an engine that cites its sources cites those
    keys — somewhere inside a shape this code deliberately does not model.
    The walk is therefore structural rather than schema-aware: every string
    in the payload (dict keys included, pydantic models dumped) is checked
    against the index, exactly first and as a substring second, so a bare key,
    a ``/path/to/key.txt``, and a ``key<SEP>key`` join all resolve.

    An engine that cites nothing simply yields ``[]`` — which is reported as
    "no provenance", never as a zero score (criterion §8.2#4).

    Args:
        payload: Anything the engine returned (mapping, sequence, model, or
            string).
        index: ``doc_key`` → message guids, from :func:`doc_guid_index` plus
            whatever aliases the adapter indexed its documents under.

    Returns:
        The cited messages' guids, deduplicated, in the order the keys were
        encountered.
    """
    keys = list(index)
    matched: list[str] = []
    seen_keys: set[str] = set()
    for text in _iter_strings(payload):
        for key in _keys_in_text(text, keys):
            if key not in seen_keys:
                seen_keys.add(key)
                matched.append(key)
    guids: list[str] = []
    seen_guids: set[str] = set()
    for key in matched:
        for guid in index[key]:
            if guid not in seen_guids:
                seen_guids.add(guid)
                guids.append(guid)
    return guids


def _by_thread(messages: Sequence[SourceMessage]) -> dict[str, list[SourceMessage]]:
    """Group messages by thread, chronologically within each thread.

    Args:
        messages: Merged messages (any order).

    Returns:
        ``thread_id`` → its messages, threads in first-appearance order of the
        sorted stream (so the grouping is deterministic, not dict-insertion
        luck).
    """
    grouped: dict[str, list[SourceMessage]] = {}
    for message in sorted(messages, key=lambda message: (message.timestamp, message.guid)):
        grouped.setdefault(message.thread_id, []).append(message)
    return grouped


def _by_day(messages: Sequence[SourceMessage]) -> dict[date, list[SourceMessage]]:
    """Group one thread's chronological messages by calendar day (UTC).

    Args:
        messages: One thread's messages, already sorted oldest first.

    Returns:
        ``date`` → that day's messages, days in chronological order.
    """
    grouped: dict[date, list[SourceMessage]] = {}
    for message in messages:
        grouped.setdefault(message.timestamp.date(), []).append(message)
    return grouped


def _render_lines(messages: Sequence[SourceMessage]) -> str:
    """Render messages as transcript lines with their reactions folded in.

    Reactions are sorted by ``(kind, sender)`` rather than left in parse
    order, so the rendered text is a pure function of the message's content —
    the property that keeps an unchanged conversation upsert-identical.

    Args:
        messages: Messages to render, in order.

    Returns:
        The rendered block, newline-joined and without a trailing newline.
    """
    lines: list[str] = []
    for message in messages:
        stamp = message.timestamp.strftime("[%Y-%m-%d %H:%M] ")
        lines.append(f"{stamp}{message.sender_name}: {message.text}")
        lines.extend(
            f"  [{tapback.sender_name} {tapback.kind} this]"
            for tapback in sorted(message.tapbacks, key=lambda t: (t.kind, t.sender_name))
        )
    return "\n".join(lines)


def _transcript_doc(thread_id: str, messages: Sequence[SourceMessage]) -> TranscriptDoc:
    """Build one transcript document from a packed run of a thread's messages.

    The participant list names whoever spoke *in this document*, not everyone
    in the thread: a document's text must depend only on its own messages, or
    a later message elsewhere in the thread would rewrite an already-indexed
    document under an unchanged key.

    Args:
        thread_id: The conversation the messages came from.
        messages: The document's messages, chronological and non-empty.

    Returns:
        The rendered document.
    """
    thread_name = messages[0].thread_name or thread_id
    participants = ", ".join(sorted({message.sender_name for message in messages}))
    first, last = messages[0].timestamp.date(), messages[-1].timestamp.date()
    span = first.isoformat() if first == last else f"{first.isoformat()}..{last.isoformat()}"
    header = f"Thread: {thread_name} (participants: {participants})"
    return TranscriptDoc(
        doc_key=f"{thread_id}::{span}",
        thread_id=thread_id,
        thread_name=thread_name,
        text=f"{header}\n\n{_render_lines(messages)}",
        message_guids=[message.guid for message in messages],
    )


def _iter_strings(payload: Any, depth: int = 0) -> Iterator[str]:
    """Walk any engine payload depth-first, yielding every string in it.

    Args:
        payload: Mapping, sequence, pydantic model, string, or scalar.
        depth: Current recursion depth (the cap guards against cycles).

    Yields:
        Each string encountered, mapping keys included.
    """
    if depth > _MAX_PAYLOAD_DEPTH:
        return
    if isinstance(payload, str):
        yield payload
    elif isinstance(payload, BaseModel):
        yield from _iter_strings(payload.model_dump(), depth + 1)
    elif isinstance(payload, Mapping):
        for key, value in payload.items():
            if isinstance(key, str):
                yield key
            yield from _iter_strings(value, depth + 1)
    elif isinstance(payload, Sequence | set | frozenset) and not isinstance(payload, bytes):
        for item in payload:
            yield from _iter_strings(item, depth + 1)


def _keys_in_text(text: str, keys: Sequence[str]) -> list[str]:
    """Find the document keys cited by one string from an engine payload.

    Args:
        text: A string from the payload.
        keys: Every indexed document key.

    Returns:
        The keys this string cites. An exact match short-circuits; otherwise
        keys are matched as substrings, dropping any key that is itself
        contained in another matched key (one render never emits a key that
        prefixes another, so such a pair means the index accumulated across
        differently-packed builds — the longer, more specific key is the one
        the engine actually indexed under).
    """
    if not text:
        return []
    if text in keys:
        return [text]
    hits = [key for key in keys if key in text]
    return [key for key in hits if not any(other != key and key in other for other in hits)]
