"""Synthetic iMessage fixture corpus for the graph eval (spec_graphrag §12).

Writes a **real** ``chat.db`` — the schema subset
:mod:`varagity.graph.sources.imessage` reads — so the bake-off and the
permanent regression harness parse the fixture through the product code path
(``find_message_source`` → ``batch_for_path``), never a test shortcut.

Two profiles (plan decision #11):

- ``smoke`` — the hand-scripted conversations only (~226 messages, 10 threads,
  5 named participants + the owner). Adapter development, harness unit tests,
  the incremental-reindex check.
- ``full`` — the same scripted messages **verbatim**, plus deterministic
  filler up to > 10,000 messages spanning a decade (the §12 Q2 bake-off
  corpus).

Golden QA (``data/eval/graph_golden_qa.jsonl``) is authored strictly against
scripted messages, which are byte-identical in both profiles, so a golden
never depends on filler. Filler is held to the opposite discipline: it must
never mention a golden entity or fact term (:data:`FILLER_BLOCKLIST`,
asserted at generation time) — that is what keeps the corpus discriminative
at 10⁴ scale rather than merely large.

The scripted content is synthetic: invented people, invented events, no real
data (spec_graphrag §6). The one exception is the ``attributedBody`` blob
reused from the Phase-1 decode fixture (innocuous, owner-reviewed), so that
the NULL-``text`` decode path runs inside the eval too.
"""

import json
import logging
import random
import re
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Repo-root-relative inputs, matching the eval module's convention
# (``EVAL_CORPUS``/``GOLDEN_PATH`` — every CLI command runs from the root).
SCRIPT_PATH = Path("tests/fixtures/graph/scripted_messages.json")
ATTRIBUTED_BODY_PATH = Path("tests/fixtures/graph/attributed_body_sample.bin")
GRAPH_GOLDEN_PATH = Path("data/eval/graph_golden_qa.jsonl")

# What ATTRIBUTED_BODY_PATH decodes to (pinned by the Phase-1 decode tests).
# A scripted message flagged ``attributed`` is written with a NULL ``text``
# column, so its scripted text must be exactly what the blob carries.
ATTRIBUTED_BODY_TEXT = "You gonna be home tonight"

# The reaction vocabulary, in ``associated_message_type`` order (2000 + index
# adds, 3000 + index removes). Deliberately re-declared here rather than
# imported from the parser's module-private tuple — the house
# "hardcode again, pin with a regression test" idiom
# (``tests/unit/test_graph_fixtures.py`` fails if the two ever diverge).
TAPBACK_KINDS: tuple[str, ...] = (
    "loved",
    "liked",
    "disliked",
    "laughed",
    "emphasized",
    "questioned",
)
_TAPBACK_ADD_BASE = 2000

# Apple's reference date and the era boundary: ``message.date`` is in seconds
# before macOS 10.13/iOS 11 and nanoseconds after, and the fixture writes
# each row in its era's units so the parser's per-row magnitude detection is
# exercised by real mixed data (spec_graphrag §10.1).
_APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=UTC)
_ERA_BOUNDARY = datetime(2017, 1, 1, tzinfo=UTC)
_NANOS_PER_SECOND = 1_000_000_000

# Only the columns the parser selects (a real chat.db has dozens more).
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

# Minimum **parsed** message count per profile (tapback rows are reactions,
# not messages, so they don't count). ``full`` clears the spec's 10⁴ bar;
# ``smoke`` adds no filler at all.
PROFILE_TARGETS: dict[str, int] = {"smoke": 0, "full": 10_001}

# Every golden entity/fact term. Filler may not contain any of them (as a
# whole word, optionally pluralised) — checked at generation time so a later
# vocabulary edit can't quietly dilute the corpus into noise that competes
# with the designed distractors.
FILLER_BLOCKLIST: tuple[str, ...] = (
    # cast
    "bob",
    "jane",
    "carol",
    "dave",
    "erin",
    "marisol",
    "petrov",
    "nakamura",
    "okafor",
    "mendez",
    "ruiz",
    "vasquez",
    # Q1/Q2 hardware opinions
    "keyboard",
    "trackpad",
    "mouse",
    "arm",
    "x86",
    "linux",
    "computer",
    "laptop",
    "desktop",
    "thinkpad",
    "ssd",
    "ram",
    "mechanical",
    "silicon",
    "chip",
    "processor",
    "monitor",
    "browser",
    # Dave's technology position
    "paper",
    "notebook",
    "pen",
    "phone",
    "notification",
    "app",
    "typewriter",
    # relations and the cabin
    "birthday",
    "teacher",
    "aunt",
    "cabin",
    "roof",
    "alternator",
    "starter motor",
    "chili",
    "firewood",
    "cooler",
    "board game",
    # food
    "bakery",
    "sourdough",
    "ramen",
    "curry",
    "croissant",
    "kimchi",
    "brine",
    # music
    "piano",
    "organ",
    "recital",
    "harpsichord",
    "chopin",
    "debussy",
    "pachelbel",
    "hymn",
    "wedding",
    "funeral",
    "scale",
)

_BLOCKED_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(term) for term in FILLER_BLOCKLIST) + r")s?\b",
    re.IGNORECASE,
)

# Filler vocabulary: deliberately banal errand/weather/logistics chatter with
# no overlap with the scripted content's topics or cast.
_FILLER_FIRST_NAMES: tuple[str, ...] = (
    "Alex",
    "Priya",
    "Marco",
    "Nia",
    "Tomas",
    "Sofia",
    "Ivan",
    "Lena",
    "Omar",
    "Rosa",
    "Kenji",
    "Ada",
    "Pablo",
    "Ingrid",
    "Hassan",
    "Mira",
    "Victor",
    "Yuki",
    "Noor",
    "Felix",
    "Greta",
    "Samir",
    "Talia",
    "Bruno",
)
_FILLER_LAST_NAMES: tuple[str, ...] = (
    "Romero",
    "Nair",
    "Silva",
    "Bergman",
    "Haddad",
    "Kimura",
    "Duarte",
    "Novak",
    "Costa",
    "Farrell",
    "Iqbal",
    "Lindqvist",
)
_FILLER_THREAD_NAMES: tuple[str, ...] = (
    "Carpool",
    "Neighbors",
    "Weekend Crew",
    "Office Lunch",
    "Volleyball",
    "Garden Swap",
    "Trivia Night",
    "Potluck",
    "Soccer Parents",
    "Rideshare",
)
_FILLER_TEMPLATES: tuple[str, ...] = (
    "running {number} minutes late, {excuse}",
    "on my way, {excuse}",
    "can you grab {item} while you are out",
    "did you remember the {item}",
    "we are out of {item} again",
    "{plan} got moved to {day}, does that still work",
    "are we still on for {day}",
    "{place} is closed until {day} apparently",
    "just left {place}, home in a bit",
    "parking at {place} is a nightmare today",
    "the {thing} is making that noise again",
    "finally fixed the {thing}",
    "{weather} out there, take a jacket",
    "it is {weather} again, unbelievable",
    "the {animal} got into the {thing} again",
    "walked the {animal} early, it was quiet out",
    "package arrived, I left it by the door",
    "laundry is in, it should be done before {day}",
    "made way too much soup, come get some",
    "tell {relative} I said hello",
    "call me when you get a chance, no rush",
    "long day. going to bed early",
    "how did {plan} go",
    "{plan} was fine, nothing exciting",
    "thanks for {day}, that was a good one",
)
_FILLER_SLOTS: dict[str, tuple[str, ...]] = {
    "number": ("five", "ten", "fifteen", "twenty"),
    "excuse": (
        "traffic on the bridge",
        "stuck at the light by the school",
        "left later than I meant to",
        "had to stop for gas",
    ),
    "item": (
        "milk",
        "eggs",
        "dish soap",
        "trash bags",
        "batteries",
        "sparkling water",
        "lemons",
        "ice",
    ),
    "plan": (
        "dinner",
        "the dentist",
        "the oil change",
        "the meeting",
        "the haircut",
        "the checkup",
    ),
    "day": (
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "tomorrow",
        "next week",
    ),
    "place": (
        "the store",
        "the gym",
        "the pharmacy",
        "the bank",
        "the car wash",
        "the library",
        "the post office",
        "the garden center",
    ),
    "thing": (
        "sink",
        "garage door",
        "back gate",
        "porch light",
        "dryer",
        "sprinkler",
        "fence",
    ),
    "weather": ("pouring rain", "freezing", "ninety degrees", "windy", "foggy"),
    "animal": ("dog", "cat", "neighbor's dog"),
    "relative": ("your mother", "the kids", "your sister", "the neighbors"),
}

# Filler shape: ~40 threads of day-bursts (the day boundaries Phase 3's
# transcript renderer splits on), 3–12 messages per burst.
_FILLER_THREAD_COUNT = 40
_FILLER_PARTICIPANTS_PER_THREAD = (2, 5)
_FILLER_BURST_SIZE = (3, 12)
_FILLER_WINDOW = (datetime(2013, 1, 1, tzinfo=UTC), datetime(2024, 12, 31, tzinfo=UTC))


class FixtureManifest(BaseModel):
    """What one :func:`build_fixture_chat_db` call wrote (never persisted).

    Returned for assertions and for the harness's settings pins: the parser
    maps handles to names through ``GRAPH_HANDLE_NAMES`` and the owner
    through ``GRAPH_OWNER_ALIASES``, so a caller that wants the scripted
    display names ("Bob Nakamura", not "+12145550101") pins
    :attr:`handle_names` and :attr:`owner_label` before parsing.

    Attributes:
        profile: The profile built (``smoke`` | ``full``).
        seed: The filler RNG seed (irrelevant for ``smoke``, recorded anyway).
        message_count: Parsed messages written (scripted + filler); tapback
            rows are reactions, not messages, and are excluded.
        scripted_count: Messages from the hand-authored script.
        filler_count: Generated filler messages (0 for ``smoke``).
        tapback_count: Reaction rows written (they fold onto their targets).
        thread_counts: Message count per thread guid.
        guids: Every message guid written, in write order — the set the
            golden QA's ``required_guids`` is validated against.
        first_timestamp: Earliest message time (aware UTC).
        last_timestamp: Latest message time (aware UTC).
        owner_label: Display name the script was authored with for
            ``is_from_me`` messages.
        handle_names: ``handle`` → display name for every participant
            (scripted and filler).
    """

    profile: str
    seed: int
    message_count: int
    scripted_count: int
    filler_count: int
    tapback_count: int
    thread_counts: dict[str, int]
    guids: list[str]
    first_timestamp: datetime
    last_timestamp: datetime
    owner_label: str
    handle_names: dict[str, str]


class _Row(BaseModel):
    """One ``message`` row queued for the writer (internal bookkeeping).

    Attributes:
        guid: The message guid (``fx-…`` scripted, ``fill-…`` filler, and
            ``…-tb<n>`` for reaction rows).
        chat_rowid: ROWID of the chat this row is joined to.
        text: The ``text`` column (``None`` for attributedBody-only and
            reaction rows).
        blob: The ``attributedBody`` column, when the row carries one.
        when: Send time (aware UTC) — converted to the era's units on write.
        is_from_me: Whether the owner sent it.
        handle_rowid: ROWID in ``handle`` for a non-owner sender.
        assoc_type: ``associated_message_type`` (0 for a normal message).
        assoc_guid: ``associated_message_guid`` (reaction rows only).
    """

    guid: str
    chat_rowid: int
    text: str | None
    blob: bytes | None = None
    when: datetime
    is_from_me: bool
    handle_rowid: int | None
    assoc_type: int = 0
    assoc_guid: str | None = None


def apple_date(when: datetime) -> int:
    """Encode a datetime as a ``message.date`` value in its era's units.

    Rows before :data:`_ERA_BOUNDARY` are written in seconds (the pre-macOS
    10.13/iOS 11 format) and later rows in nanoseconds, so a single fixture
    database contains both eras exactly as a decade-spanning real export
    does.

    Args:
        when: The send time (aware UTC).

    Returns:
        The raw ``message.date`` integer.
    """
    seconds = int((when - _APPLE_EPOCH).total_seconds())
    return seconds if when < _ERA_BOUNDARY else seconds * _NANOS_PER_SECOND


def assert_filler_clean(text: str, *, context: str) -> None:
    """Reject filler text that mentions a golden entity or fact term.

    The discipline that keeps the ``full`` profile discriminative: every
    designed distractor lives in the scripted conversations, so filler must
    be inert. Matching is whole-word (optionally pluralised) and
    case-insensitive — "warm" is not "arm".

    Args:
        text: The generated filler string (message, name, or thread label).
        context: Where it came from, for the error message.

    Raises:
        ValueError: If ``text`` contains a :data:`FILLER_BLOCKLIST` term.
    """
    hit = _BLOCKED_RE.search(text)
    if hit is not None:
        raise ValueError(
            f"filler {context} mentions the golden term {hit.group(0)!r}: {text!r} — "
            "filler must never compete with the scripted distractors"
        )


def _load_script(path: Path) -> dict[str, Any]:
    """Read and sanity-check the hand-authored conversation script.

    Args:
        path: The ``scripted_messages.json`` file.

    Returns:
        The parsed script (participants, threads, messages).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a message names an unknown thread or sender, a guid
            repeats, a tapback names an unknown kind, or an ``attributed``
            message's text is not what the captured blob decodes to.
    """
    script: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    thread_keys = {thread["key"] for thread in script["threads"]}
    sender_keys = {person["key"] for person in script["participants"]} | {"me"}
    seen: set[str] = set()
    for message in script["messages"]:
        guid = message["guid"]
        if guid in seen:
            raise ValueError(f"{path}: duplicate scripted guid {guid!r}")
        seen.add(guid)
        if message["thread"] not in thread_keys:
            raise ValueError(f"{path}: message {guid!r} names unknown thread {message['thread']!r}")
        if message["from"] not in sender_keys:
            raise ValueError(f"{path}: message {guid!r} names unknown sender {message['from']!r}")
        if message.get("attributed") and message["text"] != ATTRIBUTED_BODY_TEXT:
            raise ValueError(
                f"{path}: message {guid!r} is flagged attributed, so its text must be "
                f"{ATTRIBUTED_BODY_TEXT!r} (what the captured blob decodes to)"
            )
        for tapback in message.get("tapbacks", []):
            if tapback["kind"] not in TAPBACK_KINDS:
                raise ValueError(
                    f"{path}: message {guid!r} has unknown tapback kind "
                    f"{tapback['kind']!r} (expected one of {TAPBACK_KINDS})"
                )
            if tapback["from"] not in sender_keys:
                raise ValueError(
                    f"{path}: a tapback on {guid!r} names unknown sender {tapback['from']!r}"
                )
    return script


def _scripted_rows(
    script: dict[str, Any],
    *,
    handle_rowids: dict[str, int],
    chat_rowids: dict[str, int],
    blob: bytes,
) -> tuple[list[_Row], list[_Row]]:
    """Turn the script's messages into message rows and reaction rows.

    Args:
        script: The loaded script.
        handle_rowids: Participant key → ``handle`` ROWID.
        chat_rowids: Thread key → ``chat`` ROWID.
        blob: The captured ``attributedBody`` bytes, for ``attributed``
            messages (written with a NULL ``text``, as newer OSes do).

    Returns:
        The ``(messages, reactions)`` row pair, in script order.
    """
    messages: list[_Row] = []
    reactions: list[_Row] = []
    for entry in script["messages"]:
        sender = entry["from"]
        is_from_me = sender == "me"
        when = datetime.fromisoformat(entry["at"])
        attributed = bool(entry.get("attributed"))
        chat_rowid = chat_rowids[entry["thread"]]
        messages.append(
            _Row(
                guid=entry["guid"],
                chat_rowid=chat_rowid,
                text=None if attributed else entry["text"],
                blob=blob if attributed else None,
                when=when,
                is_from_me=is_from_me,
                handle_rowid=None if is_from_me else handle_rowids[sender],
            )
        )
        for index, tapback in enumerate(entry.get("tapbacks", [])):
            reactor = tapback["from"]
            target = entry["guid"]
            # Both real association-guid shapes: ``bp:`` targets the whole
            # message, ``p:0/`` its first part (the parser strips either).
            prefix = "bp:" if tapback.get("whole") else "p:0/"
            reactions.append(
                _Row(
                    guid=f"{target}-tb{index}",
                    chat_rowid=chat_rowid,
                    text=None,
                    when=when + timedelta(minutes=1),
                    is_from_me=reactor == "me",
                    handle_rowid=None if reactor == "me" else handle_rowids[reactor],
                    assoc_type=_TAPBACK_ADD_BASE + TAPBACK_KINDS.index(tapback["kind"]),
                    assoc_guid=f"{prefix}{target}",
                )
            )
    return messages, reactions


def _filler_people(rng: random.Random) -> list[tuple[str, str]]:
    """Invent the filler cast as ``(handle, display name)`` pairs.

    Args:
        rng: The seeded RNG (consumed in a fixed order, so the cast is
            reproducible).

    Returns:
        One pair per filler participant; handles never collide with the
        scripted cast's (different area code).

    Raises:
        ValueError: If a generated name hits :data:`FILLER_BLOCKLIST`.
    """
    people: list[tuple[str, str]] = []
    for index, first in enumerate(_FILLER_FIRST_NAMES):
        last = _FILLER_LAST_NAMES[index % len(_FILLER_LAST_NAMES)]
        name = f"{first} {last}"
        assert_filler_clean(name, context="participant name")
        people.append((f"+15125557{index:03d}", name))
    rng.shuffle(people)
    return people


def _filler_line(rng: random.Random) -> str:
    """Compose one banal filler message from the template vocabulary.

    Args:
        rng: The seeded RNG.

    Returns:
        The rendered line.

    Raises:
        ValueError: If the rendered line hits :data:`FILLER_BLOCKLIST`.
    """
    template = rng.choice(_FILLER_TEMPLATES)
    values = {slot: rng.choice(options) for slot, options in _FILLER_SLOTS.items()}
    line = template.format(**values)
    assert_filler_clean(line, context="message")
    return line


def _filler_rows(
    *,
    count: int,
    rng: random.Random,
    handle_rowids: dict[str, int],
    filler_chats: list[tuple[int, list[str]]],
) -> list[_Row]:
    """Generate ``count`` filler messages as day-bursts across the threads.

    Each pass picks a random day per thread and lays a short burst of
    messages minutes apart on it — conversation-shaped noise, so the
    transcript renderer's day splitting has real boundaries to find.

    Args:
        count: How many filler messages to generate.
        rng: The seeded RNG.
        handle_rowids: Participant handle → ``handle`` ROWID.
        filler_chats: One ``(chat ROWID, participant handles)`` per filler
            thread.

    Returns:
        The generated rows (unsorted; the parser orders by timestamp).
    """
    start, end = _FILLER_WINDOW
    span_days = (end - start).days
    rows: list[_Row] = []
    while len(rows) < count:
        for chat_rowid, handles in filler_chats:
            if len(rows) >= count:
                break
            when = start + timedelta(
                days=rng.randrange(span_days),
                hours=rng.randrange(7, 23),
                minutes=rng.randrange(60),
            )
            burst = min(rng.randint(*_FILLER_BURST_SIZE), count - len(rows))
            for _ in range(burst):
                handle = rng.choice([*handles, ""])  # "" is the owner
                rows.append(
                    _Row(
                        guid=f"fill-{len(rows):06d}",
                        chat_rowid=chat_rowid,
                        text=_filler_line(rng),
                        when=when,
                        is_from_me=handle == "",
                        handle_rowid=handle_rowids.get(handle),
                    )
                )
                when += timedelta(minutes=rng.randint(1, 9))
    return rows


def _write_db(
    dest: Path,
    *,
    handles: list[tuple[int, str]],
    chats: list[tuple[int, str, str]],
    chat_handles: list[tuple[int, int]],
    rows: list[_Row],
) -> None:
    """Write the fixture database (overwriting any previous build).

    Args:
        dest: Destination ``.db`` path (parents are created).
        handles: ``(ROWID, handle)`` pairs.
        chats: ``(ROWID, guid, display_name)`` triples.
        chat_handles: ``(chat ROWID, handle ROWID)`` membership pairs.
        rows: Every message row (messages and reactions), in write order —
            the list index becomes the ``message`` ROWID.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.unlink(missing_ok=True)
    conn = sqlite3.connect(str(dest))
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
            [
                (
                    index,
                    row.guid,
                    row.text,
                    row.blob,
                    apple_date(row.when),
                    int(row.is_from_me),
                    row.handle_rowid,
                    row.assoc_type,
                    row.assoc_guid,
                )
                for index, row in enumerate(rows, start=1)
            ],
        )
        conn.executemany(
            "INSERT INTO chat_message_join (chat_id, message_id) VALUES (?, ?)",
            [(row.chat_rowid, index) for index, row in enumerate(rows, start=1)],
        )
        conn.commit()
    finally:
        conn.close()


def build_fixture_chat_db(
    dest: Path,
    *,
    profile: str = "smoke",
    seed: int = 13,
    message_target: int | None = None,
    script_path: Path = SCRIPT_PATH,
    blob_path: Path = ATTRIBUTED_BODY_PATH,
) -> FixtureManifest:
    """Build the synthetic ``chat.db`` for a profile and describe what it holds.

    The scripted conversations are written verbatim in every profile (golden
    QA is authored against them); ``full`` appends deterministic filler until
    the message target is met. Determinism is over *parsed content*, not file
    bytes — SQLite's page layout is not the contract.

    Args:
        dest: Destination ``.db`` path; any previous build is replaced.
        profile: ``smoke`` (scripted only) or ``full`` (scripted + filler to
            > 10,000 messages).
        seed: RNG seed for the filler (same seed ⇒ same parsed content).
        message_target: Overrides the profile's minimum parsed-message count
            — the knob unit tests use to exercise ``full`` cheaply.
        script_path: The hand-authored conversation script.
        blob_path: The captured ``attributedBody`` blob for NULL-``text``
            rows.

    Returns:
        The :class:`FixtureManifest` describing the build.

    Raises:
        FileNotFoundError: If the script or blob fixture is missing.
        ValueError: If ``profile`` is unknown, the script is inconsistent, or
            generated filler mentions a golden term.
    """
    if profile not in PROFILE_TARGETS:
        raise ValueError(
            f"unknown fixture profile {profile!r}; expected one of {list(PROFILE_TARGETS)}"
        )
    target = PROFILE_TARGETS[profile] if message_target is None else message_target

    script = _load_script(script_path)
    blob = blob_path.read_bytes()
    rng = random.Random(seed)

    handle_rowids: dict[str, int] = {}
    handles: list[tuple[int, str]] = []
    handle_names: dict[str, str] = {}
    for person in script["participants"]:
        rowid = len(handles) + 1
        handle_rowids[person["key"]] = rowid
        handles.append((rowid, person["handle"]))
        handle_names[person["handle"]] = person["name"]

    chat_rowids: dict[str, int] = {}
    chat_guids: dict[int, str] = {}
    chats: list[tuple[int, str, str]] = []
    chat_handles: list[tuple[int, int]] = []
    for thread in script["threads"]:
        rowid = len(chats) + 1
        chat_rowids[thread["key"]] = rowid
        chat_guids[rowid] = thread["guid"]
        chats.append((rowid, thread["guid"], thread["display_name"]))
        chat_handles.extend((rowid, handle_rowids[key]) for key in thread["participants"])

    messages, reactions = _scripted_rows(
        script, handle_rowids=handle_rowids, chat_rowids=chat_rowids, blob=blob
    )
    scripted_count = len(messages)

    filler: list[_Row] = []
    if target > scripted_count:
        people = _filler_people(rng)
        filler_handle_rowids: dict[str, int] = {}
        for handle, name in people:
            rowid = len(handles) + 1
            filler_handle_rowids[handle] = rowid
            handles.append((rowid, handle))
            handle_names[handle] = name
        filler_chats: list[tuple[int, list[str]]] = []
        for index in range(_FILLER_THREAD_COUNT):
            rowid = len(chats) + 1
            guid = f"fx-thread-fill-{index:02d}"
            # Every third filler thread is an unnamed group, so the parser's
            # participant-name fallback runs at scale too.
            named = _FILLER_THREAD_NAMES[index % len(_FILLER_THREAD_NAMES)]
            label = "" if index % 3 == 0 else named
            if label:
                assert_filler_clean(label, context="thread name")
            chat_guids[rowid] = guid
            chats.append((rowid, guid, label))
            start = index % len(people)
            members = [
                people[(start + offset) % len(people)][0]
                for offset in range(rng.randint(*_FILLER_PARTICIPANTS_PER_THREAD))
            ]
            chat_handles.extend((rowid, filler_handle_rowids[handle]) for handle in members)
            filler_chats.append((rowid, members))
        filler = _filler_rows(
            count=target - scripted_count,
            rng=rng,
            handle_rowids=filler_handle_rowids,
            filler_chats=filler_chats,
        )

    all_messages = [*messages, *filler]
    _write_db(
        dest,
        handles=handles,
        chats=chats,
        chat_handles=chat_handles,
        rows=[*all_messages, *reactions],
    )

    thread_counts: dict[str, int] = {}
    for row in all_messages:
        guid = chat_guids[row.chat_rowid]
        thread_counts[guid] = thread_counts.get(guid, 0) + 1
    times = [row.when for row in all_messages]
    manifest = FixtureManifest(
        profile=profile,
        seed=seed,
        message_count=len(all_messages),
        scripted_count=scripted_count,
        filler_count=len(filler),
        tapback_count=len(reactions),
        thread_counts=thread_counts,
        guids=[row.guid for row in all_messages],
        first_timestamp=min(times),
        last_timestamp=max(times),
        owner_label=script["owner_name"],
        handle_names=handle_names,
    )
    logger.info(
        "built %s fixture at %s: %d message(s) (%d scripted + %d filler), %d thread(s), %s..%s",
        profile,
        dest,
        manifest.message_count,
        manifest.scripted_count,
        manifest.filler_count,
        len(thread_counts),
        manifest.first_timestamp.date(),
        manifest.last_timestamp.date(),
    )
    return manifest
