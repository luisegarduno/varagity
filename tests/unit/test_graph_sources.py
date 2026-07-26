"""Unit tests for the message-source registry and the iMessage parser (spec_graphrag §10.1).

Each test builds a throwaway ``chat.db`` in ``tmp_path`` with raw SQL (only
the columns the parser reads) — the technique Phase 2's fixture generator
generalizes. Selection is structural (``find_message_source``), so there is
no config vocabulary tuple to guard; a registry-enumeration test (the chunker
precedent) pins the vocabulary instead.
"""

import sqlite3
from collections import OrderedDict
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError
from typedstream import GenericArchivedObject, TypedValue
from typedstream.types.foundation import NSArray, NSDictionary, NSString

from varagity.config import Settings
from varagity.graph import (
    MESSAGE_SOURCE_REGISTRY,
    MessageSource,
    batch_for_path,
    find_message_source,
    get_message_source,
)
from varagity.graph.sources.imessage import IMessageSource, _first_ns_string, _main
from varagity.stores.records import content_hash, derive_doc_id

_APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=UTC)

_SCHEMA = """
CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT);
CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, guid TEXT, display_name TEXT);
CREATE TABLE chat_handle_join (chat_id INTEGER, handle_id INTEGER);
CREATE TABLE message (
    ROWID INTEGER PRIMARY KEY,
    guid TEXT,
    text TEXT,
    attributedBody BLOB,
    date INTEGER,
    is_from_me INTEGER,
    handle_id INTEGER,
    associated_message_type INTEGER,
    associated_message_guid TEXT
);
CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER);
"""

_Rows = Sequence[tuple[object, ...]]


def _seconds_epoch(when: datetime) -> int:
    """Apple seconds-epoch value for ``when`` (pre-macOS 10.13 era)."""
    return int((when - _APPLE_EPOCH).total_seconds())


def _nanos_epoch(when: datetime) -> int:
    """Apple nanoseconds-epoch value for ``when`` (post-macOS 10.13 era)."""
    return int((when - _APPLE_EPOCH).total_seconds()) * 1_000_000_000


def _message_row(
    rowid: int,
    guid: str,
    *,
    text: str | None = None,
    attributed_body: bytes | None = None,
    date: int = 0,
    is_from_me: int = 0,
    handle_id: int | None = None,
    assoc_type: int = 0,
    assoc_guid: str | None = None,
) -> tuple[object, ...]:
    """Build a ``message`` row tuple matching the parser's column order."""
    return (
        rowid,
        guid,
        text,
        attributed_body,
        date,
        is_from_me,
        handle_id,
        assoc_type,
        assoc_guid,
    )


def _build_chat_db(
    path: Path,
    *,
    handles: _Rows = (),
    chats: _Rows = (),
    chat_handles: _Rows = (),
    messages: _Rows = (),
    chat_messages: _Rows = (),
) -> Path:
    """Write a throwaway ``chat.db`` with the schema subset the parser reads."""
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(_SCHEMA)
        conn.executemany("INSERT INTO handle (ROWID, id) VALUES (?, ?)", handles)
        conn.executemany("INSERT INTO chat (ROWID, guid, display_name) VALUES (?, ?, ?)", chats)
        conn.executemany(
            "INSERT INTO chat_handle_join (chat_id, handle_id) VALUES (?, ?)", chat_handles
        )
        conn.executemany(
            "INSERT INTO message (ROWID, guid, text, attributedBody, date, is_from_me,"
            " handle_id, associated_message_type, associated_message_guid)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            messages,
        )
        conn.executemany(
            "INSERT INTO chat_message_join (chat_id, message_id) VALUES (?, ?)", chat_messages
        )
        conn.commit()
    finally:
        conn.close()
    return path


class TestRegistry:
    def test_imessage_is_registered_on_package_import(self) -> None:
        assert "imessage" in MESSAGE_SOURCE_REGISTRY
        assert isinstance(MESSAGE_SOURCE_REGISTRY["imessage"], IMessageSource)
        assert isinstance(MESSAGE_SOURCE_REGISTRY["imessage"], MessageSource)

    def test_registry_enumeration_pins_the_vocabulary(self) -> None:
        """The chunker-style guard: only ``imessage`` in v1."""
        assert sorted(MESSAGE_SOURCE_REGISTRY) == ["imessage"]

    def test_get_message_source_returns_the_registered_instance(self) -> None:
        assert get_message_source("imessage") is MESSAGE_SOURCE_REGISTRY["imessage"]

    def test_unknown_source_raises_keyerror_listing_available(self) -> None:
        with pytest.raises(KeyError) as excinfo:
            get_message_source("whatsapp")
        message = str(excinfo.value)
        assert "whatsapp" in message
        assert "imessage" in message  # the listing names what IS available

    def test_find_message_source_picks_imessage_for_a_chat_db(self, tmp_path: Path) -> None:
        db = _build_chat_db(
            tmp_path / "chat.db",
            chats=[(10, "c", "")],
            messages=[_message_row(1, "g", text="hi", is_from_me=1)],
            chat_messages=[(10, 1)],
        )
        assert find_message_source(db) is MESSAGE_SOURCE_REGISTRY["imessage"]

    def test_find_message_source_returns_none_for_a_non_message_file(self, tmp_path: Path) -> None:
        text_file = tmp_path / "notes.txt"
        text_file.write_text("not a database")
        assert find_message_source(text_file) is None

    def test_batch_for_path_raises_when_nothing_matches(self, tmp_path: Path) -> None:
        text_file = tmp_path / "notes.txt"
        text_file.write_text("nope")
        with pytest.raises(ValueError, match="no message source matches"):
            batch_for_path(text_file, tmp_path)


class TestMatches:
    def test_accepts_a_real_chat_db(self, tmp_path: Path) -> None:
        db = _build_chat_db(
            tmp_path / "chat.db",
            chats=[(10, "c", "")],
            messages=[_message_row(1, "g", text="hi", is_from_me=1)],
            chat_messages=[(10, 1)],
        )
        assert IMessageSource().matches(db) is True

    def test_rejects_a_wrong_suffix(self, tmp_path: Path) -> None:
        """A real chat.db named ``.txt`` fails the cheap suffix gate."""
        db = _build_chat_db(
            tmp_path / "chat.txt",
            messages=[_message_row(1, "g", text="hi", is_from_me=1)],
        )
        assert IMessageSource().matches(db) is False

    def test_rejects_a_non_sqlite_db(self, tmp_path: Path) -> None:
        fake = tmp_path / "fake.db"
        fake.write_bytes(b"this is not a sqlite database at all")
        assert IMessageSource().matches(fake) is False

    def test_rejects_a_sqlite_db_without_a_message_table(self, tmp_path: Path) -> None:
        other = tmp_path / "other.db"
        conn = sqlite3.connect(str(other))
        conn.execute("CREATE TABLE notes (id INTEGER)")
        conn.commit()
        conn.close()
        assert IMessageSource().matches(other) is False

    def test_rejects_a_missing_file(self, tmp_path: Path) -> None:
        assert IMessageSource().matches(tmp_path / "absent.db") is False


class TestEpochs:
    def test_both_eras_and_zero_date_decode(
        self, tmp_path: Path, settings_env: Callable[..., None]
    ) -> None:
        settings_env()  # hermetic defaults (owner label "Me")
        seconds_dt = datetime(2015, 6, 1, 12, 0, tzinfo=UTC)
        nanos_dt = datetime(2023, 3, 4, 18, 22, tzinfo=UTC)
        db = _build_chat_db(
            tmp_path / "chat.db",
            chats=[(10, "c", "")],
            messages=[
                _message_row(
                    1, "sec", text="seconds", date=_seconds_epoch(seconds_dt), is_from_me=1
                ),
                _message_row(2, "ns", text="nanos", date=_nanos_epoch(nanos_dt), is_from_me=1),
                _message_row(3, "zero", text="undated", date=0, is_from_me=1),
            ],
            chat_messages=[(10, 1), (10, 2), (10, 3)],
        )
        by_guid = {m.guid: m for m in IMessageSource().parse(db)}
        assert by_guid["sec"].timestamp == seconds_dt
        assert by_guid["ns"].timestamp == nanos_dt
        assert by_guid["zero"].timestamp == _APPLE_EPOCH

    def test_messages_are_sorted_oldest_first(
        self, tmp_path: Path, settings_env: Callable[..., None]
    ) -> None:
        settings_env()
        db = _build_chat_db(
            tmp_path / "chat.db",
            chats=[(10, "c", "")],
            messages=[
                _message_row(
                    1,
                    "new",
                    text="newer",
                    date=_nanos_epoch(datetime(2024, 1, 1, tzinfo=UTC)),
                    is_from_me=1,
                ),
                _message_row(
                    2,
                    "old",
                    text="older",
                    date=_seconds_epoch(datetime(2012, 1, 1, tzinfo=UTC)),
                    is_from_me=1,
                ),
            ],
            chat_messages=[(10, 1), (10, 2)],
        )
        assert [m.guid for m in IMessageSource().parse(db)] == ["old", "new"]


class TestIdentity:
    def test_doc_id_matches_the_recipe_and_guid_is_preserved(
        self, tmp_path: Path, settings_env: Callable[..., None]
    ) -> None:
        settings_env()
        db = _build_chat_db(
            tmp_path / "chat.db",
            chats=[(10, "c", "")],
            messages=[_message_row(1, "stable-guid-xyz", text="hello", is_from_me=1)],
            chat_messages=[(10, 1)],
        )
        batch = batch_for_path(db, tmp_path)
        assert batch.relative_path == "chat.db"
        expected = derive_doc_id("chat.db", content_hash(db.read_bytes()))
        assert batch.doc_id == expected
        assert [m.guid for m in batch.messages] == ["stable-guid-xyz"]

    def test_doc_id_falls_back_to_file_name_outside_root(
        self, tmp_path: Path, settings_env: Callable[..., None]
    ) -> None:
        settings_env()
        inside = tmp_path / "corpus"
        inside.mkdir()
        db = _build_chat_db(
            inside / "chat.db",
            chats=[(10, "c", "")],
            messages=[_message_row(1, "g", text="hi", is_from_me=1)],
            chat_messages=[(10, 1)],
        )
        other_root = tmp_path / "elsewhere"
        other_root.mkdir()
        batch = batch_for_path(db, other_root)
        assert batch.relative_path == "chat.db"  # not relative to other_root → file name
        assert batch.doc_id == derive_doc_id("chat.db", content_hash(db.read_bytes()))


class TestThreads:
    def test_named_chat_keeps_its_display_name(
        self, tmp_path: Path, settings_env: Callable[..., None]
    ) -> None:
        settings_env()
        db = _build_chat_db(
            tmp_path / "chat.db",
            chats=[(11, "c11", "Birthday Planning")],
            messages=[_message_row(1, "g", text="hi", is_from_me=1)],
            chat_messages=[(11, 1)],
        )
        assert IMessageSource().parse(db)[0].thread_name == "Birthday Planning"

    def test_empty_display_name_falls_back_to_sorted_participants(
        self, tmp_path: Path, settings_env: Callable[..., None]
    ) -> None:
        settings_env(GRAPH_HANDLE_NAMES="+15550001111=Bob,jane@example.com=Jane")
        db = _build_chat_db(
            tmp_path / "chat.db",
            handles=[(1, "+15550001111"), (2, "jane@example.com")],
            chats=[(10, "c10", "")],
            chat_handles=[(10, 1), (10, 2)],
            messages=[_message_row(1, "g", text="hey all", is_from_me=1)],
            chat_messages=[(10, 1)],
        )
        assert IMessageSource().parse(db)[0].thread_name == "Bob, Jane"

    def test_unmapped_handle_falls_back_to_the_raw_handle(
        self, tmp_path: Path, settings_env: Callable[..., None]
    ) -> None:
        settings_env(GRAPH_HANDLE_NAMES="")
        db = _build_chat_db(
            tmp_path / "chat.db",
            handles=[(3, "+15559998888")],
            chats=[(10, "c10", "")],
            chat_handles=[(10, 3)],
            messages=[_message_row(1, "g", text="yo", is_from_me=0, handle_id=3)],
            chat_messages=[(10, 1)],
        )
        message = IMessageSource().parse(db)[0]
        assert message.sender_handle == "+15559998888"
        assert message.sender_name == "+15559998888"

    def test_mapped_handle_uses_the_display_name(
        self, tmp_path: Path, settings_env: Callable[..., None]
    ) -> None:
        settings_env(GRAPH_HANDLE_NAMES="+15550001111=Bob")
        db = _build_chat_db(
            tmp_path / "chat.db",
            handles=[(1, "+15550001111")],
            chats=[(10, "c10", "")],
            chat_handles=[(10, 1)],
            messages=[_message_row(1, "g", text="hi", is_from_me=0, handle_id=1)],
            chat_messages=[(10, 1)],
        )
        message = IMessageSource().parse(db)[0]
        assert message.sender_handle == "+15550001111"
        assert message.sender_name == "Bob"

    def test_is_from_me_uses_the_owner_alias_label(
        self, tmp_path: Path, settings_env: Callable[..., None]
    ) -> None:
        settings_env(GRAPH_OWNER_ALIASES="Ego,Ego Full")
        db = _build_chat_db(
            tmp_path / "chat.db",
            chats=[(10, "c10", "")],
            messages=[_message_row(1, "g", text="mine", is_from_me=1)],
            chat_messages=[(10, 1)],
        )
        message = IMessageSource().parse(db)[0]
        assert message.is_from_me is True
        assert message.sender_name == "Ego"
        assert message.sender_handle == ""

    def test_is_from_me_defaults_to_me_without_aliases(
        self, tmp_path: Path, settings_env: Callable[..., None]
    ) -> None:
        settings_env(GRAPH_OWNER_ALIASES="")
        db = _build_chat_db(
            tmp_path / "chat.db",
            chats=[(10, "c10", "")],
            messages=[_message_row(1, "g", text="mine", is_from_me=1)],
            chat_messages=[(10, 1)],
        )
        assert IMessageSource().parse(db)[0].sender_name == "Me"


class TestTapbacks:
    def test_p_and_bp_prefixed_targets_fold_and_rows_are_not_messages(
        self, tmp_path: Path, settings_env: Callable[..., None]
    ) -> None:
        settings_env(GRAPH_HANDLE_NAMES="+1=Bob,+2=Carol")
        db = _build_chat_db(
            tmp_path / "chat.db",
            handles=[(1, "+1"), (2, "+2")],
            chats=[(10, "c10", "")],
            chat_handles=[(10, 1), (10, 2)],
            messages=[
                _message_row(1, "target", text="great idea", is_from_me=1),
                _message_row(2, "tb-loved", assoc_type=2000, assoc_guid="p:0/target", handle_id=1),
                _message_row(3, "tb-liked", assoc_type=2001, assoc_guid="bp:target", handle_id=2),
            ],
            chat_messages=[(10, 1), (10, 2), (10, 3)],
        )
        messages = IMessageSource().parse(db)
        assert {m.guid for m in messages} == {"target"}  # tapback rows are not messages
        folded = {(t.kind, t.sender_name) for t in messages[0].tapbacks}
        assert folded == {("loved", "Bob"), ("liked", "Carol")}

    def test_a_remove_cancels_a_matching_add(
        self, tmp_path: Path, settings_env: Callable[..., None]
    ) -> None:
        settings_env()
        db = _build_chat_db(
            tmp_path / "chat.db",
            chats=[(10, "c10", "")],
            messages=[
                _message_row(1, "t", text="hi", is_from_me=1),
                _message_row(2, "add", assoc_type=2000, assoc_guid="p:0/t", is_from_me=1),
                _message_row(3, "remove", assoc_type=3000, assoc_guid="p:0/t", is_from_me=1),
            ],
            chat_messages=[(10, 1), (10, 2), (10, 3)],
        )
        assert IMessageSource().parse(db)[0].tapbacks == []

    def test_an_orphan_tapback_is_dropped_without_error(
        self, tmp_path: Path, settings_env: Callable[..., None]
    ) -> None:
        settings_env()
        db = _build_chat_db(
            tmp_path / "chat.db",
            chats=[(10, "c10", "")],
            messages=[
                _message_row(1, "real", text="hi", is_from_me=1),
                _message_row(
                    2, "orphan", assoc_type=2000, assoc_guid="p:0/NOT-PRESENT", is_from_me=1
                ),
            ],
            chat_messages=[(10, 1), (10, 2)],
        )
        messages = IMessageSource().parse(db)
        assert {m.guid for m in messages} == {"real"}
        assert messages[0].tapbacks == []

    def test_a_malformed_association_guid_is_ignored(
        self, tmp_path: Path, settings_env: Callable[..., None]
    ) -> None:
        settings_env()
        db = _build_chat_db(
            tmp_path / "chat.db",
            chats=[(10, "c10", "")],
            messages=[
                _message_row(1, "real", text="hi", is_from_me=1),
                _message_row(
                    2, "malformed", assoc_type=2000, assoc_guid="p:no-slash", is_from_me=1
                ),
            ],
            chat_messages=[(10, 1), (10, 2)],
        )
        messages = IMessageSource().parse(db)
        assert {m.guid for m in messages} == {"real"}
        assert messages[0].tapbacks == []


class TestAttributedBody:
    """Decode of NULL-``text`` rows whose body lives in the typedstream blob.

    The two round-trip tests need a real ``attributedBody`` blob captured in
    this phase's manual R3 gate (``tests/fixtures/graph/attributed_body_hello.bin``,
    innocuous owner-reviewed content); they skip until it lands. The decode
    *logic* is unit-covered below without needing a real export.
    """

    _FIXTURE = Path(__file__).parent.parent / "fixtures" / "graph" / "attributed_body_hello.bin"

    def test_attributed_body_row_decodes(
        self, tmp_path: Path, settings_env: Callable[..., None]
    ) -> None:
        pytest.skip("fixture pending real-export capture")
        settings_env()
        blob = self._FIXTURE.read_bytes()
        db = _build_chat_db(
            tmp_path / "chat.db",
            chats=[(10, "c10", "")],
            messages=[_message_row(1, "g", text=None, attributed_body=blob, is_from_me=1)],
            chat_messages=[(10, 1)],
        )
        messages = IMessageSource().parse(db)
        assert len(messages) == 1
        assert messages[0].text  # the captured innocuous content, decoded

    def test_corrupt_attributed_body_row_is_skipped_and_counted(
        self, tmp_path: Path, settings_env: Callable[..., None]
    ) -> None:
        pytest.skip("fixture pending real-export capture")
        settings_env()
        blob = self._FIXTURE.read_bytes()
        db = _build_chat_db(
            tmp_path / "chat.db",
            chats=[(10, "c10", "")],
            messages=[
                _message_row(1, "good", text=None, attributed_body=blob, is_from_me=1),
                _message_row(2, "bad", text=None, attributed_body=b"corrupt blob", is_from_me=1),
            ],
            chat_messages=[(10, 1), (10, 2)],
        )
        assert {m.guid for m in IMessageSource().parse(db)} == {"good"}

    def test_decode_helper_returns_none_on_empty_or_garbage(self) -> None:
        """The failure mode the corrupt-row skip relies on (no real fixture needed)."""
        assert IMessageSource._decode_attributed_body(None) is None
        assert IMessageSource._decode_attributed_body(b"") is None
        assert IMessageSource._decode_attributed_body(b"not a typedstream at all") is None

    def test_first_ns_string_walks_the_decoded_object_tree(self) -> None:
        """The message text is the first NSString in DFS order across containers."""
        text = NSString()
        text.value = "the real message"
        nested = NSDictionary()
        nested.contents = OrderedDict({"attr": TypedValue(b"@", text)})
        array = NSArray()
        array.elements = [nested]
        obj = GenericArchivedObject(
            clazz=None, super_object=None, contents=[TypedValue(b"@", array)]
        )
        assert _first_ns_string(obj) == "the real message"
        # super_object, plain list, and plain dict branches
        via_super = GenericArchivedObject(clazz=None, super_object=text, contents=[])
        assert _first_ns_string(via_super) == "the real message"
        assert _first_ns_string([1, "x", {"k": text}]) == "the real message"
        # leaves with no recoverable NSString
        assert _first_ns_string(42) is None
        non_string = NSString()
        non_string.value = 123  # not a str — must not be returned
        assert _first_ns_string(non_string) is None

    def test_null_text_row_with_undecodable_body_is_skipped(
        self, tmp_path: Path, settings_env: Callable[..., None]
    ) -> None:
        """Parse-level skip path (the else-branch into decode), no real fixture needed."""
        settings_env()
        db = _build_chat_db(
            tmp_path / "chat.db",
            chats=[(10, "c10", "")],
            messages=[
                _message_row(1, "good", text="hello", is_from_me=1),
                _message_row(2, "bad", text=None, attributed_body=b"garbage", is_from_me=1),
            ],
            chat_messages=[(10, 1), (10, 2)],
        )
        assert {m.guid for m in IMessageSource().parse(db)} == {"good"}


class TestMainTool:
    def test_reports_a_summary_and_zero_exit(
        self,
        tmp_path: Path,
        settings_env: Callable[..., None],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        settings_env()
        db = _build_chat_db(
            tmp_path / "chat.db",
            chats=[(10, "c10", "Group")],
            messages=[
                _message_row(
                    1,
                    "a",
                    text="first",
                    date=_seconds_epoch(datetime(2014, 1, 1, tzinfo=UTC)),
                    is_from_me=1,
                ),
                _message_row(
                    2,
                    "b",
                    text="second",
                    date=_nanos_epoch(datetime(2022, 1, 1, tzinfo=UTC)),
                    is_from_me=1,
                ),
                _message_row(3, "tb", assoc_type=2000, assoc_guid="p:0/a", is_from_me=1),
            ],
            chat_messages=[(10, 1), (10, 2), (10, 3)],
        )
        code = _main([str(db)])
        out = capsys.readouterr().out
        assert code == 0
        assert "messages: 2" in out
        assert "date range:" in out
        assert "reactions=" in out  # message "a" carries the folded tapback

    def test_bad_usage_returns_two(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert _main([]) == 2
        assert "usage:" in capsys.readouterr().err

    def test_non_chat_db_returns_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        not_db = tmp_path / "notes.txt"
        not_db.write_text("nope")
        assert _main([str(not_db)]) == 1
        assert "not an iMessage chat.db" in capsys.readouterr().err

    def test_empty_database_reports_no_messages(
        self, tmp_path: Path, settings_env: Callable[..., None], capsys: pytest.CaptureFixture[str]
    ) -> None:
        settings_env()
        db = _build_chat_db(tmp_path / "chat.db")
        assert _main([str(db)]) == 0
        assert "no messages parsed" in capsys.readouterr().out


class TestGraphConfig:
    def test_owner_alias_list_strips_and_dedupes_in_order(self) -> None:
        settings = Settings(_env_file=None, GRAPH_OWNER_ALIASES=" John , John Doe ,John")
        assert settings.graph_owner_alias_list == ["John", "John Doe"]

    def test_owner_label_is_the_first_alias(self) -> None:
        assert Settings(_env_file=None, GRAPH_OWNER_ALIASES="Ego,Other").graph_owner_label == "Ego"

    def test_owner_label_defaults_to_me(self) -> None:
        assert Settings(_env_file=None, GRAPH_OWNER_ALIASES="").graph_owner_label == "Me"

    def test_handle_name_map_parses_pairs(self) -> None:
        settings = Settings(
            _env_file=None, GRAPH_HANDLE_NAMES="+15551234567=Bob, jane@x.com = Jane "
        )
        assert settings.graph_handle_name_map == {"+15551234567": "Bob", "jane@x.com": "Jane"}

    def test_handle_names_rejects_an_entry_without_equals(self) -> None:
        with pytest.raises(ValidationError, match="GRAPH_HANDLE_NAMES"):
            Settings(_env_file=None, GRAPH_HANDLE_NAMES="+15551234567=Bob,justaname")

    def test_handle_name_map_reads_a_contacts_file(self, tmp_path: Path) -> None:
        contacts = tmp_path / "contacts.txt"
        contacts.write_text(
            "+12145550101=Bob Loblaw,\n+12285550102=Jane NCAS,\n+12145550103=Carol (Poker),\n",
            encoding="utf-8",
        )
        settings = Settings(_env_file=None, GRAPH_HANDLE_NAMES_FILE=str(contacts))
        assert settings.graph_handle_name_map == {
            "+12145550101": "Bob Loblaw",
            "+12285550102": "Jane NCAS",
            "+12145550103": "Carol (Poker)",
        }

    def test_inline_pairs_override_file_pairs(self, tmp_path: Path) -> None:
        contacts = tmp_path / "contacts.txt"
        contacts.write_text("+15551234567=File Name\n+15550000000=Keep Me\n", encoding="utf-8")
        settings = Settings(
            _env_file=None,
            GRAPH_HANDLE_NAMES_FILE=str(contacts),
            GRAPH_HANDLE_NAMES="+15551234567=Inline Name",
        )
        assert settings.graph_handle_name_map == {
            "+15551234567": "Inline Name",
            "+15550000000": "Keep Me",
        }

    def test_missing_contacts_file_raises_a_clear_error(self, tmp_path: Path) -> None:
        settings = Settings(_env_file=None, GRAPH_HANDLE_NAMES_FILE=str(tmp_path / "nope.txt"))
        with pytest.raises(ValueError, match="GRAPH_HANDLE_NAMES_FILE"):
            _ = settings.graph_handle_name_map

    def test_malformed_contacts_file_entry_names_the_file_and_entry(self, tmp_path: Path) -> None:
        contacts = tmp_path / "contacts.txt"
        contacts.write_text("+15551234567=Bob\nnot-a-pair\n", encoding="utf-8")
        settings = Settings(_env_file=None, GRAPH_HANDLE_NAMES_FILE=str(contacts))
        with pytest.raises(ValueError, match="not-a-pair"):
            _ = settings.graph_handle_name_map

    def test_defaults(self) -> None:
        settings = Settings(_env_file=None)
        assert settings.GRAPH_DOCS_PATH == "./graph-docs"
        assert settings.GRAPH_HANDLE_NAMES_FILE == ""
        assert settings.graph_owner_alias_list == []
        assert settings.graph_handle_name_map == {}
