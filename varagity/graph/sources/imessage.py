"""The iMessage ``chat.db`` message source (spec_graphrag §10.1).

Parses a copied iMessage SQLite store into :class:`SourceMessage` objects
with full fidelity — the structure today's prose parsers throw away:

- **Stable identity** is ``message.guid`` (never the volatile ``ROWID``), so
  an overlapping re-ingest of a growing database upserts rather than
  duplicates.
- **Both Apple epoch eras** are decoded per row by magnitude: ``message.date``
  counts from 2001-01-01 UTC, in **seconds** pre-macOS 10.13/iOS 11 and
  **nanoseconds** after — a decade-spanning database contains both.
- **``attributedBody``** is decoded when the plain ``text`` column is NULL
  (the newer-OS default): the body lives in a serialized
  ``NSAttributedString`` typedstream blob, decoded here via ``pytypedstream``.
- **Tapbacks** (``associated_message_type`` 2000–2005 add / 3000–3005 remove)
  fold onto the message they react to; they never become standalone messages.
- **Ego is structural**: ``is_from_me`` marks the owner's messages, mapped to
  ``settings.graph_owner_label`` with an empty handle.
- **Handles → names**: participant handles (phone/email) are mapped through
  ``settings.graph_handle_name_map``, falling back to the raw handle.

The database *is* the export: copy it **with its ``-wal``/``-shm`` sidecars**
or recent messages are missing (SQLite honors the sidecars automatically when
they sit beside the file). The parser opens the file read-only and never
writes.

Running the module is the R3 verification tool
(``uv run python -m varagity.graph.sources.imessage /path/to/chat.db``): it
prints message/thread counts, the date range, tapback-fold count, and a few
sample messages so a real export can be sanity-checked by hand.
"""

import logging
import sqlite3
import sys
from collections import defaultdict
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import typedstream
from typedstream import GenericArchivedObject, TypedGroup
from typedstream.types.foundation import NSArray, NSDictionary, NSString

from varagity.config import get_settings
from varagity.graph.sources.base import SourceMessage, Tapback, register

logger = logging.getLogger(__name__)

# SQLite's fixed 16-byte file header (the magic sniff in ``matches``).
_SQLITE_MAGIC = b"SQLite format 3\x00"

# Apple's reference date: 2001-01-01 00:00:00 UTC.
_APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=UTC)
# ``message.date`` magnitudes above this are nanoseconds (post-10.13/iOS 11),
# at or below are seconds. 10**12 ns is ~16 minutes past the epoch and 10**12 s
# is the year ~33670 — no real message falls between, so per-row magnitude
# detection is unambiguous across the decade-spanning database (§12 Q2).
_NS_THRESHOLD = 10**12
_NANOS_PER_SECOND = 1_000_000_000

# Tapback association-type ranges: 2000–2005 add a reaction, 3000–3005 remove
# one; the offset into this table is the reaction kind.
_TAPBACK_ADD_BASE = 2000
_TAPBACK_REMOVE_BASE = 3000
_TAPBACK_KINDS = ("loved", "liked", "disliked", "laughed", "emphasized", "questioned")

# Only the columns the parser reads (real ``chat.db`` has dozens more).
_MESSAGE_QUERY = """
    SELECT
        m.ROWID, m.guid, m.text, m.attributedBody, m.date,
        m.is_from_me, m.handle_id,
        m.associated_message_type, m.associated_message_guid,
        cmj.chat_id
    FROM message AS m
    JOIN chat_message_join AS cmj ON cmj.message_id = m.ROWID
    ORDER BY m.date, m.ROWID
"""


def _connect_ro(path: Path) -> sqlite3.Connection:
    """Open a SQLite database read-only via a percent-encoded file URI.

    Read-only mode never writes to the file (or its WAL), and the ``as_uri``
    encoding keeps paths with spaces/special characters valid in the URI.

    Args:
        path: The database file (assumed to exist — callers probe first).

    Returns:
        A read-only connection.
    """
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


@register("imessage")
class IMessageSource:
    """Parses an iMessage ``chat.db`` into normalized messages (spec_graphrag §10.1)."""

    def matches(self, path: Path) -> bool:
        """Report whether ``path`` is an iMessage ``chat.db`` (a cheap probe).

        Three gates, cheapest first: a ``.db`` suffix, the 16-byte SQLite
        magic header, and a ``message`` table in ``sqlite_master``. Any read
        or SQLite error means "not my format" — the probe never raises.

        Args:
            path: Candidate file.

        Returns:
            ``True`` only if all three gates pass.
        """
        if path.suffix.lower() != ".db":
            return False
        try:
            with path.open("rb") as handle:
                magic = handle.read(len(_SQLITE_MAGIC))
        except OSError:
            return False
        if magic != _SQLITE_MAGIC:
            return False
        try:
            conn = _connect_ro(path)
            try:
                row = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='message'"
                ).fetchone()
            finally:
                conn.close()
        except sqlite3.Error:
            return False
        return row is not None

    def parse(self, path: Path, *, verbose: int = 0) -> list[SourceMessage]:
        """Parse the database into messages, oldest first (spec_graphrag §10.1).

        Reads handles, threads, and messages in three passes, folds tapbacks
        onto their targets, and skips rows with no recoverable body (counted
        in the summary log, never emitted blank). A message linked into more
        than one chat is emitted once (its first thread).

        Args:
            path: The ``chat.db`` to parse (already accepted by
                :meth:`matches`).
            verbose: Validated console verbosity (0–2); the parse summary is
                a ``logging`` line independent of it (the R3 tool prints the
                human-facing view).

        Returns:
            Every recoverable message, sorted by timestamp then guid.
        """
        settings = get_settings()
        owner_label = settings.graph_owner_label
        name_map = settings.graph_handle_name_map

        conn = _connect_ro(path)
        try:
            handle_by_rowid: dict[int, str] = {
                rowid: handle for rowid, handle in conn.execute("SELECT ROWID, id FROM handle")
            }

            def name_for_handle(handle_rowid: int | None) -> tuple[str, str]:
                """Map a handle ROWID to its ``(raw_handle, display_name)`` pair."""
                handle = handle_by_rowid.get(handle_rowid, "") if handle_rowid else ""
                return handle, name_map.get(handle, handle)

            chat_guid: dict[int, str] = {}
            chat_display: dict[int, str] = {}
            for chat_rowid, guid, display_name in conn.execute(
                "SELECT ROWID, guid, display_name FROM chat"
            ):
                chat_guid[chat_rowid] = guid
                chat_display[chat_rowid] = display_name or ""
            participants: dict[int, set[str]] = defaultdict(set)
            for chat_id, handle_id in conn.execute(
                "SELECT chat_id, handle_id FROM chat_handle_join"
            ):
                participants[chat_id].add(name_for_handle(handle_id)[1])
            # Group chats usually have an empty display_name — fall back to
            # the sorted participant names (spec_graphrag §10.1).
            thread_name_by_chat: dict[int, str] = {
                chat_rowid: (
                    display.strip()
                    or ", ".join(sorted(n for n in participants.get(chat_rowid, set()) if n))
                )
                for chat_rowid, display in chat_display.items()
            }

            rows = conn.execute(_MESSAGE_QUERY).fetchall()
        finally:
            conn.close()

        messages_by_guid: dict[str, SourceMessage] = {}
        tapback_net: dict[tuple[str, str, str], int] = defaultdict(int)
        seen_rowids: set[int] = set()
        skipped = 0

        for (
            rowid,
            guid,
            text,
            attributed_body,
            date,
            is_from_me,
            handle_id,
            assoc_type,
            assoc_guid,
            chat_id,
        ) in rows:
            if rowid in seen_rowids:  # a message linked into >1 chat — emit once
                continue
            seen_rowids.add(rowid)

            reaction = self._tapback_kind(assoc_type)
            if reaction is not None:
                kind, is_add = reaction
                target = self._strip_target_prefix(assoc_guid)
                if target is None:  # malformed association guid
                    continue
                reactor = owner_label if is_from_me else name_for_handle(handle_id)[1]
                tapback_net[(target, reactor, kind)] += 1 if is_add else -1
                continue

            body = (
                text if (text and text.strip()) else self._decode_attributed_body(attributed_body)
            )
            if not body:  # nothing recoverable — count and skip, never emit blank
                skipped += 1
                continue
            if is_from_me:
                sender_handle, sender_name = "", owner_label
            else:
                sender_handle, sender_name = name_for_handle(handle_id)
            messages_by_guid[guid] = SourceMessage(
                guid=guid,
                thread_id=chat_guid.get(chat_id, ""),
                thread_name=thread_name_by_chat.get(chat_id, ""),
                sender_handle=sender_handle,
                sender_name=sender_name,
                is_from_me=bool(is_from_me),
                timestamp=self._apple_timestamp(date),
                text=body,
            )

        folded = self._fold_tapbacks(tapback_net, messages_by_guid)
        messages = sorted(messages_by_guid.values(), key=lambda m: (m.timestamp, m.guid))
        logger.info(
            "parsed %s: %d message(s) decoded, %d skipped, %d tapback(s) folded",
            path.name,
            len(messages),
            skipped,
            folded,
        )
        return messages

    @staticmethod
    def _apple_timestamp(raw: int | None) -> datetime:
        """Decode a ``message.date`` into a tz-aware UTC datetime.

        Detects the epoch era per row by magnitude (:data:`_NS_THRESHOLD`);
        a ``0``/NULL date resolves to the epoch base rather than being
        dropped (dropping would silently lose rows, and goldens never depend
        on undated ones). Integer arithmetic keeps the decode exact.

        Args:
            raw: The raw ``message.date`` value (``None`` or ``0`` allowed).

        Returns:
            The send time as an aware UTC datetime.
        """
        if not raw:
            return _APPLE_EPOCH
        if raw > _NS_THRESHOLD:
            seconds, nanos = divmod(int(raw), _NANOS_PER_SECOND)
            return _APPLE_EPOCH + timedelta(seconds=seconds, microseconds=nanos // 1000)
        return _APPLE_EPOCH + timedelta(seconds=int(raw))

    @staticmethod
    def _tapback_kind(assoc_type: int | None) -> tuple[str, bool] | None:
        """Classify an ``associated_message_type`` as a tapback.

        Args:
            assoc_type: The row's ``associated_message_type`` (``0``/NULL for
                a normal message).

        Returns:
            A ``(kind, is_add)`` pair for a tapback add (2000–2005) or remove
            (3000–3005), or ``None`` when the row is not a tapback.
        """
        if assoc_type is None:
            return None
        span = len(_TAPBACK_KINDS)
        if _TAPBACK_ADD_BASE <= assoc_type < _TAPBACK_ADD_BASE + span:
            return _TAPBACK_KINDS[assoc_type - _TAPBACK_ADD_BASE], True
        if _TAPBACK_REMOVE_BASE <= assoc_type < _TAPBACK_REMOVE_BASE + span:
            return _TAPBACK_KINDS[assoc_type - _TAPBACK_REMOVE_BASE], False
        return None

    @staticmethod
    def _strip_target_prefix(assoc_guid: str | None) -> str | None:
        """Strip an ``associated_message_guid`` down to the bare target guid.

        iMessage prefixes the target with ``p:N/`` (message part N) or
        ``bp:`` (whole message); a bare guid is passed through defensively.

        Args:
            assoc_guid: The raw ``associated_message_guid``.

        Returns:
            The bare target guid, or ``None`` when the value is empty or a
            malformed ``p:`` prefix with no ``/``.
        """
        if not assoc_guid:
            return None
        if assoc_guid.startswith("bp:"):
            return assoc_guid[len("bp:") :]
        if assoc_guid.startswith("p:"):
            _, _, rest = assoc_guid.partition("/")
            return rest or None
        return assoc_guid

    @staticmethod
    def _fold_tapbacks(
        net: dict[tuple[str, str, str], int],
        messages_by_guid: dict[str, SourceMessage],
    ) -> int:
        """Fold net-positive reactions onto their target messages.

        A remove cancels a matching add per ``(target, reactor, kind)`` (the
        net count); a reaction whose target is absent from this parse is
        dropped silently (targets outside a partial export are expected).

        Args:
            net: Net add-minus-remove count per ``(target_guid, reactor,
                kind)``.
            messages_by_guid: The decoded messages, keyed by guid (mutated
                in place — reactions append to each target's ``tapbacks``).

        Returns:
            The number of reactions folded onto a present target.
        """
        folded = 0
        for (target_guid, reactor, kind), count in net.items():
            if count <= 0:
                continue
            target = messages_by_guid.get(target_guid)
            if target is None:  # reaction to a message outside this export
                continue
            target.tapbacks.append(Tapback(kind=kind, sender_name=reactor))
            folded += 1
        return folded

    @staticmethod
    def _decode_attributed_body(blob: bytes | None) -> str | None:
        """Decode a NULL-``text`` row's body from its ``attributedBody`` blob.

        Newer macOS/iOS store the body only in a serialized
        ``NSAttributedString`` typedstream (spec_graphrag §10.1). The archived
        string sits first in the object graph (attribute runs follow it), so
        a depth-first search for the first ``NSString`` value returns the
        message text. Any decode failure is a skip, not an abort (a bad blob
        must not fail the whole file); R3's real-export gate validates this
        path, with ``imessage-exporter`` as the documented contingency.

        Args:
            blob: The row's ``attributedBody`` bytes, or ``None``.

        Returns:
            The decoded body, or ``None`` when the blob is empty or
            undecodable.
        """
        if not blob:
            return None
        try:
            unarchived = typedstream.unarchive_from_data(blob)
        except Exception:  # any malformed blob skips the row (never aborts the file)
            logger.warning("attributedBody decode failed — skipping the row", exc_info=True)
            return None
        return _first_ns_string(unarchived)


def _first_ns_string(obj: object) -> str | None:
    """Depth-first search a decoded typedstream for the first ``NSString`` value.

    The message text is the first archived string in an
    ``NSAttributedString`` graph; attribute-run dictionaries (fonts, colors,
    message-part keys) follow it, so DFS order returns the body.

    Args:
        obj: A decoded typedstream node (object, container, or leaf).

    Returns:
        The first ``NSString``/``NSMutableString`` value, or ``None``.
    """
    # A real NSString carries its text in ``value``; a generic object whose
    # *super* is a known NSString also passes this isinstance check (the
    # library's "known part" proxy) but has no ``value`` of its own — fall
    # through to the child walk (which yields its super/contents) in that case.
    if isinstance(obj, NSString):
        value = getattr(obj, "value", None)
        if isinstance(value, str):
            return value
    for child in _children(obj):
        found = _first_ns_string(child)
        if found is not None:
            return found
    return None


def _children(obj: object) -> Iterator[object]:
    """Yield the child nodes of a decoded typedstream node, for :func:`_first_ns_string`.

    Args:
        obj: A decoded typedstream node.

    Yields:
        Each contained value, in archive order (nothing for a leaf).
    """
    if isinstance(obj, GenericArchivedObject):
        if obj.super_object is not None:
            yield obj.super_object
        yield from obj.contents
    elif isinstance(obj, TypedGroup):  # TypedValue is a single-value TypedGroup
        yield from obj.values
    elif isinstance(obj, NSArray):
        yield from obj.elements
    elif isinstance(obj, NSDictionary):
        yield from obj.contents.keys()
        yield from obj.contents.values()
    elif isinstance(obj, list | tuple):
        yield from obj
    elif isinstance(obj, dict):
        yield from obj.keys()
        yield from obj.values()


def _main(args: Sequence[str]) -> int:
    """R3 verification tool: parse a real ``chat.db`` and print a summary.

    Usage: ``uv run python -m varagity.graph.sources.imessage <chat.db>``.
    Prints counts, date range, tapback folds, and sample messages so a real
    export can be sanity-checked by hand (the skipped/decode-failure count
    rides the parser's own ``INFO`` summary line).

    Args:
        args: Command-line arguments after the module name (one path).

    Returns:
        Process exit code: ``0`` success, ``1`` not a ``chat.db``, ``2`` bad
        usage.
    """
    if len(args) != 1:
        print("usage: python -m varagity.graph.sources.imessage <chat.db>", file=sys.stderr)
        return 2
    path = Path(args[0])
    source = IMessageSource()
    if not source.matches(path):
        print(f"{path} is not an iMessage chat.db", file=sys.stderr)
        return 1
    messages = source.parse(path, verbose=2)
    if not messages:
        print("no messages parsed")
        return 0
    threads = {m.thread_id for m in messages}
    folded = sum(len(m.tapbacks) for m in messages)
    from_me = sum(1 for m in messages if m.is_from_me)
    print(f"messages: {len(messages)}")
    print(f"threads:  {len(threads)}")
    print(
        f"date range: {messages[0].timestamp.isoformat()} .. {messages[-1].timestamp.isoformat()}"
    )
    print(f"tapbacks folded: {folded}")
    print(f"from me: {from_me} | from others: {len(messages) - from_me}")
    count = len(messages)
    picks = sorted({0, count // 4, count // 2, (3 * count) // 4, count - 1})
    print("\nsamples:")
    for index in picks:
        message = messages[index]
        reactions = f"  reactions={[t.kind for t in message.tapbacks]}" if message.tapbacks else ""
        print(
            f"  [{message.timestamp.date()}] {message.sender_name} "
            f"@ {message.thread_name!r}: {message.text[:80]!r}{reactions}"
        )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(_main(sys.argv[1:]))
