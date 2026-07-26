"""Message-source protocol and registry (spec_graphrag §5.2, §10.2; the spec §5.1 pattern).

Each message-source module defines one implementation decorated with
``@register("name")``; unlike the chat/retriever registries (settings-field
selection), sources are chosen by **structural dispatch** — the parser
precedent (``varagity.ingest.loader._BUCKET_PARSERS``): a file is routed to
the first source whose :meth:`MessageSource.matches` accepts it, so there is
no ``config.py`` vocabulary tuple to keep in lockstep. v1 registers only
``imessage``; adding a platform later is one new module plus its import line
in :mod:`varagity.graph.sources`, zero caller edits.

A message source produces **structured messages** (speaker, timestamp,
thread, per-message identity) — not prose for chunking — so the family lives
under :mod:`varagity.graph`, not :mod:`varagity.ingest`. The models are
pydantic (not dataclasses) because the fixture generator, eval results, and
stage-2 wire payloads all serialize them.
"""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from varagity.stores.records import content_hash, derive_doc_id


class Tapback(BaseModel):
    """One folded reaction on a message (a sentiment signal, not a message).

    Tapback rows in ``chat.db`` never become standalone
    :class:`SourceMessage` records — they fold onto the message they react to
    (spec_graphrag §10.1).

    Attributes:
        kind: The reaction, one of ``loved`` | ``liked`` | ``disliked`` |
            ``laughed`` | ``emphasized`` | ``questioned``.
        sender_name: Display name of whoever reacted (the owner label when
            the reaction is the owner's).
    """

    kind: str
    sender_name: str


class SourceMessage(BaseModel):
    """One normalized message from any message source (spec_graphrag §10.2).

    Attributes:
        guid: Stable per-message identity — ``message.guid`` for iMessage,
            never the volatile ``ROWID`` (a rebuilt/merged database keeps the
            guid). Sources without native ids hash thread+sender+time+text.
        thread_id: Stable thread identity (``chat.guid`` for iMessage).
        thread_name: Human-facing thread label: the chat's ``display_name``
            when set, else the sorted participant names (group chats often
            have an empty ``display_name``).
        sender_handle: Raw handle (phone number / email); ``""`` for the
            owner (whose handle is not modelled — ego is structural).
        sender_name: Mapped display name (``GRAPH_HANDLE_NAMES``), falling
            back to the raw handle; the owner label when :attr:`is_from_me`.
        is_from_me: Whether the owner sent it (``message.is_from_me`` — the
            structural ego marker that Q3-class attribution needs).
        timestamp: Send time as a timezone-aware UTC datetime (both Apple
            epoch eras decoded to the same wall clock).
        text: The decoded message body; never empty (rows with no
            recoverable text are skipped at parse time, not emitted blank).
        tapbacks: Reactions folded onto this message (empty when none).
    """

    guid: str
    thread_id: str
    thread_name: str
    sender_handle: str
    sender_name: str
    is_from_me: bool
    timestamp: datetime
    text: str
    tapbacks: list[Tapback] = []


class MessageBatch(BaseModel):
    """A parsed source file: its stable identity plus every message in it.

    The identity recipe is the corpus recipe reused verbatim
    (:func:`varagity.stores.records.derive_doc_id` over the path relative to
    ``GRAPH_DOCS_PATH`` plus the file's byte hash), so a graph document's
    ``doc_id`` is portable across host/container/machines exactly as a RAG
    document's is.

    Attributes:
        doc_id: Stable id of the source file (see :func:`batch_for_path`).
        relative_path: POSIX path of the file relative to the graph root.
        messages: Every parsed message, oldest first.
    """

    doc_id: str
    relative_path: str
    messages: list[SourceMessage]


@runtime_checkable
class MessageSource(Protocol):
    """Parses one file format into :class:`SourceMessage` objects.

    ``runtime_checkable`` mirrors :class:`~varagity.chat.base.ChatEngine`:
    the protocol will appear in stage-2 Prefect flow signatures, and Prefect
    builds a pydantic parameter schema from the annotations at decoration
    time, which requires types usable with ``isinstance``.
    """

    def matches(self, path: Path) -> bool:
        """Report whether this source can parse ``path`` (a cheap probe).

        Args:
            path: Candidate file.

        Returns:
            ``True`` if the file is this source's format. Must never raise —
            a probe that can't read the file returns ``False``.
        """
        ...

    def parse(self, path: Path, *, verbose: int = 0) -> list[SourceMessage]:
        """Parse ``path`` into normalized messages, oldest first.

        Called only after :meth:`matches` has accepted ``path``.

        Args:
            path: File to parse (its format already confirmed by
                :meth:`matches`).
            verbose: Validated console verbosity (0–2).

        Returns:
            Every recoverable message, oldest first (unrecoverable rows are
            counted and skipped, never emitted with empty text).
        """
        ...


MESSAGE_SOURCE_REGISTRY: dict[str, MessageSource] = {}


def register[T: type[Any]](name: str) -> Callable[[T], T]:
    """Class decorator registering a message-source instance under ``name``.

    Args:
        name: Registry key (also the source's stable identifier).

    Returns:
        The decorator, which instantiates and registers the class unchanged.
    """

    def deco(cls: T) -> T:
        MESSAGE_SOURCE_REGISTRY[name] = cls()
        return cls

    return deco


def get_message_source(name: str) -> MessageSource:
    """Look up a registered message source by name.

    Args:
        name: Registry key (e.g. ``"imessage"``).

    Returns:
        The registered source instance.

    Raises:
        KeyError: If no source is registered under ``name`` (the message
            lists the available ones).
    """
    if name not in MESSAGE_SOURCE_REGISTRY:
        raise KeyError(
            f"Unknown message source {name!r}. Available: {list(MESSAGE_SOURCE_REGISTRY)}"
        )
    return MESSAGE_SOURCE_REGISTRY[name]


def find_message_source(path: Path) -> MessageSource | None:
    """Find the first registered source whose :meth:`MessageSource.matches` accepts ``path``.

    Structural dispatch (the ``_BUCKET_PARSERS`` precedent): a file with no
    matching source returns ``None`` rather than raising, so callers can
    count-and-skip an unsupported drop-in the way ingest discovery does.

    Args:
        path: Candidate file.

    Returns:
        The matching source, or ``None`` when no registered source claims it.
    """
    for source in MESSAGE_SOURCE_REGISTRY.values():
        if source.matches(path):
            return source
    return None


def batch_for_path(path: Path, root: Path, *, verbose: int = 0) -> MessageBatch:
    """Parse ``path`` into a :class:`MessageBatch` with a stable ``doc_id``.

    ``doc_id`` is derived from the path relative to ``root`` plus the file's
    byte hash — the same recipe as RAG documents
    (:func:`varagity.stores.records.derive_doc_id`) — so overlapping
    re-ingests of a growing database can upsert against a stable id. A
    ``path`` outside ``root`` (e.g. a symlink target) falls back to the file
    name, exactly as :func:`varagity.ingest.loader._ingest_file` does.

    Args:
        path: File to parse (must be claimed by a registered source).
        root: Corpus root the ``doc_id`` is relativized against (call sites
            pass ``settings.GRAPH_DOCS_PATH``).
        verbose: Validated console verbosity (0–2).

    Returns:
        The parsed batch: identity plus its messages.

    Raises:
        ValueError: If no registered source matches ``path``.
    """
    source = find_message_source(path)
    if source is None:
        raise ValueError(
            f"no message source matches {path} — registered: {list(MESSAGE_SOURCE_REGISTRY)}"
        )
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:  # target outside the corpus root (e.g. a symlink)
        relative = path.name
    file_hash = content_hash(path.read_bytes())
    doc_id = derive_doc_id(relative, file_hash)
    messages = source.parse(path, verbose=verbose)
    return MessageBatch(doc_id=doc_id, relative_path=relative, messages=messages)
