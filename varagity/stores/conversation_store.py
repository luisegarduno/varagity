"""Conversation persistence over PostgreSQL (spec_v2 §4.4, §9.1).

Single-user chat history lives in the same Postgres as the chunks (one
datastore to operate) but in independent tables: an assistant turn's
evidence is **snapshotted** into ``message_sources.trace`` — content,
context, source provenance, and the
:class:`~varagity.stores.records.RetrievalTrace` — with ``chunk_id`` kept
only as a soft reference, so a historical conversation still explains
itself after a reingest replaces the chunk rows.

The tables are created by the migration runner
(:mod:`varagity.stores.migrate`), not ``schema.sql``.
"""

import logging
import uuid
from collections.abc import Sequence
from datetime import datetime
from types import TracebackType
from typing import Any

import psycopg
from psycopg.types.json import Json
from pydantic import BaseModel

from varagity.models.llm import LLMClient, clean_response
from varagity.stores.records import RetrievedChunk
from varagity.stores.vector_store import default_conninfo

logger = logging.getLogger(__name__)

DEFAULT_TITLE = "New conversation"
"""Title given at creation; :meth:`ConversationStore.auto_title` replaces it."""

# Auto-titling prompt (spec_v2 §4.2: "LLM one-liner, cheap").
_TITLE_PROMPT = (
    "Write a title of at most six words for a conversation that starts with "
    "this question. Reply with the title only — no quotes, no punctuation at "
    "the end.\n\nQUESTION: {question}\nTITLE:"
)
_TITLE_MAX_CHARS = 80

_LIST_CONVERSATIONS_SQL = """
SELECT c.conversation_id, c.title, c.created_at, c.updated_at,
       count(m.message_id) AS message_count
FROM conversations c
LEFT JOIN messages m ON m.conversation_id = c.conversation_id
GROUP BY c.conversation_id
ORDER BY c.updated_at DESC
"""

_SELECT_MESSAGES_SQL = """
SELECT message_id, role, content, created_at, retrieval_method, latency_ms, reasoning
FROM messages
WHERE conversation_id = %s
ORDER BY created_at, message_id
"""

_SELECT_SOURCES_SQL = """
SELECT s.message_id, s.rank, s.chunk_id, s.trace
FROM message_sources s
JOIN messages m ON m.message_id = s.message_id
WHERE m.conversation_id = %s
ORDER BY s.message_id, s.rank
"""

# Newest rows first so LIMIT bounds the read in SQL; the method reverses
# to oldest-first. The message_id tiebreak matches _SELECT_MESSAGES_SQL,
# so history and transcript agree on within-timestamp order.
_RECENT_TURNS_SQL = """
SELECT role, content
FROM messages
WHERE conversation_id = %s
ORDER BY created_at DESC, message_id DESC
LIMIT %s
"""


class ConversationSummary(BaseModel):
    """One conversation as listed in the sidebar (spec_v2 §4.2).

    Attributes:
        conversation_id: The app-generated id.
        title: Current title (auto-generated or the creation default).
        created_at: Creation timestamp.
        updated_at: Last-turn timestamp (list ordering key).
        message_count: Number of persisted messages (user + assistant).
    """

    conversation_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int


class StoredSource(BaseModel):
    """One snapshotted evidence row of an assistant turn (spec_v2 §9.1).

    Attributes:
        rank: Final rank in the answer's evidence (1-based).
        chunk_id: Soft reference to the chunk that produced the snapshot.
        trace: The persisted snapshot: ``score``, ``content``, ``context``,
            source provenance fields, and the serialized
            :class:`~varagity.stores.records.RetrievalTrace` under
            ``"trace"`` (``None`` when the retriever attached none).
    """

    rank: int
    chunk_id: str
    trace: dict[str, Any]


class MessageRecord(BaseModel):
    """One persisted message, with the assistant turn's provenance.

    Attributes:
        message_id: The app-generated id.
        role: ``"user"`` or ``"assistant"``.
        content: The question or the generated answer.
        created_at: Persistence timestamp.
        retrieval_method: Retrieval method that produced an assistant turn
            (``None`` for user turns).
        latency_ms: Per-stage timings of an assistant turn (``None`` for
            user turns).
        reasoning: Captured ``<think>`` stream, if any.
        sources: The turn's snapshotted evidence, rank order.
    """

    message_id: str
    role: str
    content: str
    created_at: datetime
    retrieval_method: str | None = None
    latency_ms: dict[str, Any] | None = None
    reasoning: str | None = None
    sources: list[StoredSource] = []


class ConversationDetail(BaseModel):
    """A full transcript: the conversation row plus its messages.

    Attributes:
        conversation_id: The app-generated id.
        title: Current title.
        created_at: Creation timestamp.
        updated_at: Last-turn timestamp.
        messages: All messages, oldest first, each with its sources.
    """

    conversation_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageRecord]


def _source_snapshot(chunk: RetrievedChunk) -> dict[str, Any]:
    """Build the ``message_sources.trace`` snapshot for one chunk.

    Carries everything the provenance panel renders (spec_v2 §4.6) so the
    stored history is self-contained: score, texts, source provenance, and
    the retrieval trace.

    Args:
        chunk: The retrieved chunk backing one evidence row.

    Returns:
        The JSONB-ready snapshot dict.
    """
    return {
        "score": chunk.score,
        "content": chunk.content,
        "context": chunk.context,
        "source": chunk.metadata.get("source"),
        "file_name": chunk.metadata.get("file_name"),
        "file_type": chunk.metadata.get("file_type"),
        "page": chunk.metadata.get("page"),
        "extraction": chunk.metadata.get("extraction"),
        "trace": None if chunk.trace is None else chunk.trace.model_dump(mode="json"),
    }


class ConversationStore:
    """Conversation CRUD over PostgreSQL.

    Owns one autocommit connection (per-turn writes group into explicit
    transactions), mirroring
    :class:`~varagity.stores.vector_store.ContextualVectorDB`. Use as a
    context manager or call :meth:`close` when done.
    """

    def __init__(self, conninfo: str | None = None) -> None:
        """Connect to the configured PostgreSQL instance.

        Args:
            conninfo: libpq connection string; defaults to the
                ``POSTGRES_*`` settings.

        Raises:
            psycopg.OperationalError: If the database is unreachable.
        """
        self._conn = psycopg.connect(conninfo or default_conninfo(), autocommit=True)

    def close(self) -> None:
        """Close the underlying connection (idempotent)."""
        if not self._conn.closed:
            self._conn.close()

    def __enter__(self) -> "ConversationStore":
        """Enter a context that closes the connection on exit.

        Returns:
            This store.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the connection on context exit.

        Args:
            exc_type: Exception type, if the block raised.
            exc: Exception instance, if the block raised.
            tb: Traceback, if the block raised.
        """
        self.close()

    def create_conversation(self, title: str | None = None) -> ConversationSummary:
        """Insert a new conversation.

        Args:
            title: Initial title; defaults to :data:`DEFAULT_TITLE` (the
                first chat turn replaces it via :meth:`auto_title`).

        Returns:
            The created conversation's summary (``message_count = 0``).
        """
        conversation_id = uuid.uuid4().hex
        row = self._conn.execute(
            "INSERT INTO conversations (conversation_id, title) VALUES (%s, %s) "
            "RETURNING created_at, updated_at",
            (conversation_id, title or DEFAULT_TITLE),
        ).fetchone()
        assert row is not None  # RETURNING on a successful INSERT
        return ConversationSummary(
            conversation_id=conversation_id,
            title=title or DEFAULT_TITLE,
            created_at=row[0],
            updated_at=row[1],
            message_count=0,
        )

    def conversation_exists(self, conversation_id: str) -> bool:
        """Check whether a conversation id is known.

        Args:
            conversation_id: The id to look up.

        Returns:
            ``True`` if a matching row exists.
        """
        row = self._conn.execute(
            "SELECT 1 FROM conversations WHERE conversation_id = %s", (conversation_id,)
        ).fetchone()
        return row is not None

    def list_conversations(self) -> list[ConversationSummary]:
        """List every conversation, most recently updated first.

        Returns:
            Summaries with message counts (spec_v2 §4.2).
        """
        rows = self._conn.execute(_LIST_CONVERSATIONS_SQL).fetchall()
        return [
            ConversationSummary(
                conversation_id=row[0],
                title=row[1],
                created_at=row[2],
                updated_at=row[3],
                message_count=int(row[4]),
            )
            for row in rows
        ]

    def get_conversation(self, conversation_id: str) -> ConversationDetail | None:
        """Fetch a full transcript: messages plus snapshotted sources.

        Args:
            conversation_id: The conversation to fetch.

        Returns:
            The transcript, or ``None`` for an unknown id.
        """
        conversation_row = self._conn.execute(
            "SELECT title, created_at, updated_at FROM conversations WHERE conversation_id = %s",
            (conversation_id,),
        ).fetchone()
        if conversation_row is None:
            return None
        sources_by_message: dict[str, list[StoredSource]] = {}
        for message_id, rank, chunk_id, trace in self._conn.execute(
            _SELECT_SOURCES_SQL, (conversation_id,)
        ):
            sources_by_message.setdefault(message_id, []).append(
                StoredSource(rank=rank, chunk_id=chunk_id, trace=trace)
            )
        messages = [
            MessageRecord(
                message_id=row[0],
                role=row[1],
                content=row[2],
                created_at=row[3],
                retrieval_method=row[4],
                latency_ms=row[5],
                reasoning=row[6],
                sources=sources_by_message.get(row[0], []),
            )
            for row in self._conn.execute(_SELECT_MESSAGES_SQL, (conversation_id,))
        ]
        return ConversationDetail(
            conversation_id=conversation_id,
            title=conversation_row[0],
            created_at=conversation_row[1],
            updated_at=conversation_row[2],
            messages=messages,
        )

    def recent_turns(self, conversation_id: str, limit: int) -> list[tuple[str, str]]:
        """Fetch a conversation's newest turns, re-ordered oldest first.

        The chat engine's history feed (spec_v3 §4.4) — deliberately not
        :meth:`get_conversation`, which hydrates every message's snapshotted
        sources and traces: evidence the condenser must not see, on the
        pre-first-token hot path, growing with conversation length. Here the
        bound is applied **in SQL** (newest ``limit`` rows) and only
        ``role``/``content`` are read. Primitive pairs keep
        ``varagity/stores/`` from importing ``varagity/chat/`` — callers map
        them to their own turn type.

        Args:
            conversation_id: The conversation to read (an unknown id yields
                the empty list — existence is the caller's check).
            limit: Maximum turns returned; non-positive yields the empty
                list without touching the database.

        Returns:
            Up to ``limit`` newest ``(role, content)`` pairs, oldest first.
        """
        if limit <= 0:
            return []
        rows = self._conn.execute(_RECENT_TURNS_SQL, (conversation_id, limit)).fetchall()
        return [(row[0], row[1]) for row in reversed(rows)]

    def delete_conversation(self, conversation_id: str) -> int:
        """Delete a conversation; messages and sources cascade (spec_v2 §9.1).

        Args:
            conversation_id: The conversation to delete (unknown id is a
                no-op).

        Returns:
            The number of ``conversations`` rows deleted (0 or 1).
        """
        cursor = self._conn.execute(
            "DELETE FROM conversations WHERE conversation_id = %s", (conversation_id,)
        )
        return cursor.rowcount

    def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        *,
        retrieval_method: str | None = None,
        latency_ms: dict[str, Any] | None = None,
        reasoning: str | None = None,
        sources: Sequence[RetrievedChunk] = (),
    ) -> str:
        """Persist one turn, snapshotting an assistant turn's evidence.

        One transaction covers the message, its ``message_sources`` rows,
        and the conversation's ``updated_at`` bump — a partial failure
        leaves no dangling turn.

        Args:
            conversation_id: The conversation the turn belongs to.
            role: ``"user"`` or ``"assistant"``.
            content: The question or the generated answer.
            retrieval_method: Retrieval method used (assistant turns).
            latency_ms: Per-stage timings (assistant turns).
            reasoning: Captured ``<think>`` stream, if any.
            sources: The answer's retrieved chunks, best first; each is
                snapshotted via the spec_v2 §9.1 trace blob.

        Returns:
            The new message's id.

        Raises:
            psycopg.errors.ForeignKeyViolation: If ``conversation_id`` is
                unknown (callers validate first; the FK is the backstop).
        """
        message_id = uuid.uuid4().hex
        with self._conn.transaction():
            self._conn.execute(
                "INSERT INTO messages (message_id, conversation_id, role, content, "
                "retrieval_method, latency_ms, reasoning) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    message_id,
                    conversation_id,
                    role,
                    content,
                    retrieval_method,
                    None if latency_ms is None else Json(latency_ms),
                    reasoning,
                ),
            )
            for rank, chunk in enumerate(sources, start=1):
                self._conn.execute(
                    "INSERT INTO message_sources (message_id, rank, chunk_id, trace) "
                    "VALUES (%s, %s, %s, %s)",
                    (message_id, rank, chunk.chunk_id, Json(_source_snapshot(chunk))),
                )
            self._conn.execute(
                "UPDATE conversations SET updated_at = now() WHERE conversation_id = %s",
                (conversation_id,),
            )
        return message_id

    def auto_title(
        self, conversation_id: str, question: str, *, llm: LLMClient | None = None
    ) -> str | None:
        """Title a still-default-titled conversation from its first question.

        A cheap LLM one-liner (spec_v2 §4.2); on any generation failure the
        title falls back to the truncated question — titling must never
        break a chat turn. Conversations already titled (auto or manually)
        are left alone.

        Args:
            conversation_id: The conversation to title.
            question: The first user question.
            llm: Chat client; resolved via the model registry when omitted.

        Returns:
            The new title, or ``None`` when the conversation is unknown or
            no longer carries :data:`DEFAULT_TITLE`.
        """
        row = self._conn.execute(
            "SELECT title FROM conversations WHERE conversation_id = %s", (conversation_id,)
        ).fetchone()
        if row is None or row[0] != DEFAULT_TITLE:
            return None
        title = _generate_title(question, llm=llm)
        self._conn.execute(
            "UPDATE conversations SET title = %s WHERE conversation_id = %s",
            (title, conversation_id),
        )
        return title


def _generate_title(question: str, *, llm: LLMClient | None = None) -> str:
    """Generate a short conversation title, falling back to the question.

    Args:
        question: The first user question.
        llm: Chat client; resolved via the model registry when omitted.

    Returns:
        A non-empty title of at most ``_TITLE_MAX_CHARS`` characters.
    """
    from varagity.models.registry import get_model  # deferred: avoids an import cycle

    fallback = question.strip()[:_TITLE_MAX_CHARS] or DEFAULT_TITLE
    try:
        client = llm if llm is not None else get_model("default")
        raw = client.generate(
            [{"role": "user", "content": _TITLE_PROMPT.format(question=question)}],
            max_tokens=2048,  # reasoning models may think before the one-liner
            verbose=0,
        )
        title = clean_response(raw).strip().strip('"').strip()
        title = " ".join(title.split())  # collapse newlines/runs of spaces
        return title[:_TITLE_MAX_CHARS] if title else fallback
    except Exception:  # titling must never break the turn — any failure falls back
        logger.warning("auto-title generation failed; falling back to the question")
        return fallback
