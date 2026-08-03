"""The graph bake-off harness — ``eval graph`` (spec_graphrag §12, §8.2).

The fourth eval subcommand, and the one that decides ADR-017: build the
synthetic message corpus, run every selected registered graph engine over it
sequentially, and score each engine's *answers* — because a graph engine
returns prose, not chunk ids.

Structurally this is :func:`varagity.eval.evaluate.run_chat_eval` again
(registry-enumerated engine loop, pinned settings, fact-anchored scoring,
:func:`~varagity.eval.evaluate.write_results` persistence, a thin Prefect
shell in :mod:`varagity.pipeline.eval_flow`) with three differences that come
from the subject matter:

* **No stores, no testcontainers.** Graph engines self-store in their own
  working directories under ``data/eval/graph/`` (plan decision #9), so the
  only external needs are the live llama.cpp and infinity services.
* **Scoring is over answers.** Each golden entry carries ``expected_facts``
  (an AND of OR-groups, matched case-insensitively as substrings of the
  answer — :func:`~varagity.eval.evaluate.resolve_golden_by_fact`'s trick
  applied to prose) and ``required_guids`` (provenance). An engine that
  surfaces no source ids scores ``None`` provenance rather than 0: that
  honesty *is* criterion §8.2#4 data. Question ``kind`` slices the report and
  is never shown to an engine (plan decision #10).
* **The parser's naming settings are pinned from the fixture manifest.**
  Sender display names come from ``GRAPH_OWNER_ALIASES`` /
  ``GRAPH_HANDLE_NAMES``, so a parse without those pins yields raw phone
  handles and every golden naming "Bob" fails against a corpus that only ever
  says "+12145550101". :func:`manifest_settings_pins` derives them from what
  the generator actually wrote, and :func:`pinned_graph_settings` applies them
  the way the rest of the harness pins settings (environment export + cache
  clear — see :func:`varagity.eval.evaluate.pinned_eval_settings`).

Three cost controls shape the run, because a full-profile index is hours per
engine on a single-slot llama.cpp: ``--engine`` runs one engine per session,
``--skip-build`` reuses the corpus *and* the engines' working directories so
scoring can be iterated without re-paying an index, and ``--message-target``
caps the corpus below its profile's size — Graphiti indexes at ~51 s/message,
which makes an uncapped 10,001-message build about six days, so its
full-profile seat is a capped slice and the cap is reported with the numbers.
A capped run is a **different corpus**: :func:`corpus_stem` gives it its own
database, manifest, and per-engine working directories (``full-mt1000``), so a
capped run can never clobber the uncapped corpus another engine is still
indexing against. The scripted messages are written in every corpus regardless
of the cap, so the golden set's provenance anchors always resolve.

The **incremental re-index check** (criterion §8.2#3) runs only on the
``smoke`` profile and only when building: a second ``build`` over an
overlapping batch — the same messages plus
:data:`INCREMENTAL_DELTA_MESSAGES` new ones — measures what a re-export of a
grown ``chat.db`` costs. The delta messages are generator filler, which is
blocklist-guaranteed never to mention a golden term, so the check cannot
perturb the scores that follow it.

Engine libraries stay call-time imports (plan decision #8): selecting an
engine whose library is missing raises a clear error naming the ``bakeoff``
dependency group, and nothing here needs an engine installed to import.
"""

import logging
import os
import shutil
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from varagity.config import get_settings
from varagity.debug.show import check_verbose
from varagity.eval.datasets import GRAPH_GOLDEN_KINDS, GraphGoldenEntry, load_graph_golden
from varagity.eval.evaluate import RESULTS_DIR, write_results
from varagity.eval.graph_fixtures import (
    GRAPH_GOLDEN_PATH,
    PROFILE_TARGETS,
    FixtureManifest,
    build_fixture_chat_db,
)
from varagity.graph.base import GraphSession

# Importing the *package* is what self-registers the adapters (the registry
# idiom); varagity.graph.base alone would leave the registry empty. It costs
# nothing — no adapter imports its engine library at module level.
from varagity.graph.engines import GRAPH_ENGINE_REGISTRY, get_graph_engine
from varagity.graph.records import BuildReport, GraphStats
from varagity.graph.sources.base import MessageBatch, batch_for_path

logger = logging.getLogger(__name__)

# Everything the harness builds lives here (gitignored): ``corpus/`` holds the
# generated fixture databases, ``<profile>/<engine>/`` each engine's
# self-stored working directory (plan decision #9).
GRAPH_EVAL_ROOT = Path("data/eval/graph")

# Written beside every generated corpus so ``--skip-build`` can reuse a corpus
# without regenerating it: the manifest is what pins the parser's naming
# settings and validates the golden set's provenance anchors, and it is not
# recoverable from the database alone.
MANIFEST_SUFFIX = ".manifest.json"

# The generator's default filler seed, restated so the results document can
# record what produced the corpus (plan decision #11: same seed + profile ⇒
# identical parsed content).
FIXTURE_SEED = 13

# How many **new** messages the incremental re-index check adds on top of the
# already-indexed corpus. Small on purpose: the measurement is the *marginal*
# cost of a re-export (criterion §8.2#3), and at ~36 s/message for the slowest
# engine a larger delta would dominate the smoke run's wall clock.
INCREMENTAL_DELTA_MESSAGES = 10

# Distribution name behind each registered adapter, for the version stamp in
# the results document. Deliberately re-declared here rather than imported
# from the adapters (which name their libraries only inside ``session()``, so
# that importing them stays free) — the house "hardcode again, pin with a
# regression test" idiom; ``tests/unit/test_graph_eval.py`` fails if this map
# and the registry ever diverge.
ENGINE_DISTRIBUTIONS: dict[str, str] = {
    "cognee": "cognee",
    "graphiti": "graphiti-core",
    "lightrag": "lightrag-hku",
}


@contextmanager
def pinned_graph_settings(**pins: str) -> Iterator[None]:
    """Apply graph settings pins for a block, restoring them on exit.

    The graph counterpart of
    :func:`varagity.eval.evaluate.pinned_eval_settings`, and the same
    mechanism for the same reason: deep code (here the iMessage parser)
    resolves ``get_settings()`` internally, and the eval harness is a
    single-threaded CLI path where an environment override is safe. There is
    no standing pin set — every pin is derived from the fixture manifest the
    run just built (:func:`manifest_settings_pins`), because the names the
    corpus was written with are the only correct ones.

    Args:
        **pins: Settings to export for this block (e.g.
            ``GRAPH_OWNER_ALIASES="Sam"``).

    Yields:
        Nothing; the pinned settings are active inside the block.
    """
    saved = {name: os.environ.get(name) for name in pins}
    os.environ.update(pins)
    get_settings.cache_clear()
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        get_settings.cache_clear()


def manifest_settings_pins(*manifests: FixtureManifest) -> dict[str, str]:
    """Derive the parser's naming pins from what the generator wrote.

    Sender display names are a settings concern
    (:attr:`~varagity.config.Settings.graph_handle_name_map`,
    :attr:`~varagity.config.Settings.graph_owner_label`), so parsing a fixture
    without these pins yields raw handles and every golden question naming a
    participant is unanswerable. ``GRAPH_HANDLE_NAMES_FILE`` is pinned empty
    alongside them: a host ``.env`` pointing it at a contacts file that does
    not exist on this machine would otherwise fail the parse.

    Args:
        *manifests: The manifests of every corpus this run will parse (the
            base corpus, plus the incremental delta corpus when the check
            runs). Their handle maps are merged, later manifests winning.

    Returns:
        The pins, ready for :func:`pinned_graph_settings`.

    Raises:
        ValueError: If no manifest is given, or two manifests disagree about
            the owner label (a mismatch would silently split the owner into
            two graph entities).
    """
    if not manifests:
        raise ValueError("manifest_settings_pins needs at least one manifest")
    labels = {manifest.owner_label for manifest in manifests}
    if len(labels) > 1:
        raise ValueError(f"corpora disagree about the owner label: {sorted(labels)}")
    handle_names: dict[str, str] = {}
    for manifest in manifests:
        handle_names.update(manifest.handle_names)
    return {
        "GRAPH_OWNER_ALIASES": manifests[0].owner_label,
        "GRAPH_HANDLE_NAMES": ",".join(
            f"{handle}={name}" for handle, name in sorted(handle_names.items())
        ),
        "GRAPH_HANDLE_NAMES_FILE": "",
    }


def corpus_stem(profile: str, message_target: int | None = None) -> str:
    """Name the corpus (and the engines' workdir tier) for one run.

    A ``--message-target`` run indexes a *different corpus* from its profile's
    uncapped one, so it may not share either the generated database or the
    engines' working directories: overwriting ``full.db`` while another
    engine's multi-day full-profile build is still running against it would
    corrupt that run's provenance silently, and reusing ``full/<engine>/``
    would score a capped run against an uncapped graph.

    Args:
        profile: ``smoke`` or ``full``.
        message_target: The ``--message-target`` override, or ``None`` for the
            profile's own size.

    Returns:
        The file stem for the corpus and its manifest, and the directory name
        holding each engine's workdir: ``full`` uncapped, ``full-mt1000``
        capped.
    """
    return profile if message_target is None else f"{profile}-mt{message_target}"


def prepare_corpus(
    corpus_dir: Path,
    *,
    profile: str,
    seed: int = FIXTURE_SEED,
    message_target: int | None = None,
    name: str | None = None,
    reuse: bool = False,
) -> tuple[Path, FixtureManifest]:
    """Build (or reuse) one fixture corpus and return it with its manifest.

    Args:
        corpus_dir: Directory holding the generated databases (created if
            absent).
        profile: ``smoke`` or ``full`` (see
            :data:`~varagity.eval.graph_fixtures.PROFILE_TARGETS`).
        seed: Filler RNG seed.
        message_target: Overrides the profile's message count (the
            incremental delta corpus uses it to ask for "the corpus plus a
            few").
        name: File stem for the database and its manifest; defaults to
            ``profile``.
        reuse: Reuse an existing database **and** its manifest instead of
            regenerating (``--skip-build``). A missing or unreadable manifest
            falls back to building, since without it the run cannot pin the
            parser's naming settings.

    Returns:
        The ``(database path, manifest)`` pair.

    Raises:
        FileNotFoundError: If the scripted-message or blob fixture is missing.
        ValueError: If ``profile`` is unknown or the script is inconsistent.
    """
    stem = name or profile
    db_path = corpus_dir / f"{stem}.db"
    manifest_path = corpus_dir / f"{stem}{MANIFEST_SUFFIX}"
    if reuse and db_path.is_file() and manifest_path.is_file():
        manifest = FixtureManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        logger.info(
            "reusing the %s fixture corpus at %s (%d message(s))",
            stem,
            db_path,
            manifest.message_count,
        )
        return db_path, manifest
    if reuse:
        logger.info("no reusable %s corpus beside %s — building it", stem, manifest_path)
    manifest = build_fixture_chat_db(
        db_path, profile=profile, seed=seed, message_target=message_target
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return db_path, manifest


def validate_golden_against_manifest(
    entries: Sequence[GraphGoldenEntry], manifest: FixtureManifest
) -> None:
    """Check that every golden provenance anchor exists in the built corpus.

    The golden-validation precedent
    (:func:`varagity.eval.evaluate.validate_golden_against_store`), and strict
    for the same reason: a ``required_guid`` that no longer exists would score
    every engine's provenance down for a drift in the *fixture*, silently.
    Runs before any engine opens a session, so the failure costs seconds
    rather than hours.

    Args:
        entries: The loaded golden entries.
        manifest: The built corpus's manifest.

    Raises:
        ValueError: If any ``required_guids`` entry is absent from the corpus.
    """
    available = set(manifest.guids)
    missing = sorted(
        {guid for entry in entries for guid in entry.required_guids if guid not in available}
    )
    if missing:
        raise ValueError(
            f"{len(missing)} golden provenance anchor(s) are not in the {manifest.profile} "
            f"corpus ({manifest.message_count} messages): {missing} — the golden set and the "
            "fixture generator have drifted apart"
        )


def match_fact_groups(answer: str, expected_facts: Sequence[Sequence[str]]) -> list[bool]:
    """Report which of an entry's OR-groups the answer satisfies.

    ``expected_facts`` is an AND of OR-groups (spec_graphrag §12): a group is
    satisfied when any of its variants appears in the answer, matched
    case-insensitively as a substring — the same rule the chunk-RAG sweep uses
    against chunk text, applied to prose.

    Args:
        answer: The engine's answer text.
        expected_facts: The entry's fact groups.

    Returns:
        One boolean per group, in group order.
    """
    haystack = answer.lower()
    return [any(variant.lower() in haystack for variant in group) for group in expected_facts]


def provenance_recall(surfaced: Sequence[str], required: Sequence[str]) -> float | None:
    """Score how much of an entry's required provenance the engine surfaced.

    Args:
        surfaced: Message guids the engine's evidence carried.
        required: The entry's ``required_guids`` (non-empty by schema).

    Returns:
        ``|surfaced ∩ required| / |required|``, or ``None`` when the engine
        surfaced no guids at all — reported honestly as "this engine cannot
        cite messages" rather than as a zero score (criterion §8.2#4).

    Raises:
        ValueError: If ``required`` is empty (nothing to divide by).
    """
    if not required:
        raise ValueError("provenance_recall needs at least one required guid")
    if not surfaced:
        return None
    hits = set(surfaced) & set(required)
    return len(hits) / len(required)


def _summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Average one slice of query records into the reported numbers.

    Args:
        records: The slice's per-query records (non-empty).

    Returns:
        The slice summary: query count, mean fact recall, mean provenance
        recall over the queries that reported any (``None`` when none did),
        how many did, mean latency, and the error count.
    """
    provenance = [
        record["provenance_recall"] for record in records if record["provenance_recall"] is not None
    ]
    latencies = [record["latency_s"] for record in records if record["latency_s"] is not None]
    return {
        "n_queries": len(records),
        "fact_recall": round(sum(record["fact_recall"] for record in records) / len(records), 4),
        "provenance_recall": (round(sum(provenance) / len(provenance), 4) if provenance else None),
        "n_provenance_reported": len(provenance),
        "mean_latency_s": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "errors": sum(1 for record in records if record["error"] is not None),
    }


def measure_graph_engine(
    session: GraphSession,
    entries: Sequence[GraphGoldenEntry],
    *,
    mode: str | None = None,
    verbose: int = 0,
) -> dict[str, Any]:
    """Ask one open engine session every golden question and score the answers.

    Questions are put verbatim — an entry's ``kind`` is report metadata and is
    never shown to an engine (plan decision #10). A query that raises is
    recorded as a failed query (zero fact recall, ``None`` provenance) rather
    than aborting the engine: at hours per index, losing a whole engine's
    numbers to one bad question would be the expensive kind of strict, and
    engine failure counts are themselves criterion §8.2#2 data — the same
    posture :attr:`~varagity.graph.records.BuildReport.failures` takes for
    indexing.

    Args:
        session: The open engine session to interrogate.
        entries: The golden questions, in report order.
        mode: Engine query mode override (``None`` uses each adapter's primary
            mode — the ``--mode`` escape hatch of plan decision #13).
        verbose: Validated console verbosity (0–2).

    Returns:
        ``{"summary": …, "by_kind": …, "queries": […]}`` — the overall slice,
        one slice per :data:`~varagity.eval.datasets.GRAPH_GOLDEN_KINDS` that
        has questions, and the full per-query detail (answer text included,
        for the ADR autopsy).

    Raises:
        ValueError: If ``entries`` is empty.
    """
    if not entries:
        raise ValueError("measure_graph_engine needs at least one golden entry")
    records: list[dict[str, Any]] = []
    for entry in entries:
        required = set(entry.required_guids)
        record: dict[str, Any] = {
            "query": entry.query,
            "kind": entry.kind,
            "mode": mode,
            "latency_s": None,
            "fact_recall": 0.0,
            "missed_facts": [list(group) for group in entry.expected_facts],
            "provenance_recall": None,
            "required_guids": list(entry.required_guids),
            "matched_guids": [],
            "evidence": None,
            "answer": "",
            "error": None,
        }
        try:
            answer = session.query(entry.query, mode=mode, verbose=verbose)
        except Exception as exc:  # broad on purpose: an engine failure is §8.2#2 data
            logger.warning("query failed for %r: %s", entry.query, exc)
            record["error"] = f"{type(exc).__name__}: {exc}"
            records.append(record)
            continue
        matched = match_fact_groups(answer.answer, entry.expected_facts)
        evidence = answer.evidence
        surfaced = [guid for guid in evidence.message_guids if guid in required]
        record.update(
            {
                "mode": answer.mode,
                "latency_s": round(answer.latency_s, 2),
                "fact_recall": round(sum(matched) / len(matched), 4),
                "missed_facts": [
                    list(group)
                    for group, hit in zip(entry.expected_facts, matched, strict=True)
                    if not hit
                ],
                "provenance_recall": provenance_recall(
                    evidence.message_guids, entry.required_guids
                ),
                "matched_guids": surfaced,
                "evidence": {
                    "entities": len(evidence.entities),
                    "relations": len(evidence.relations),
                    "communities": len(evidence.communities),
                    "message_guids": len(evidence.message_guids),
                },
                "answer": answer.answer,
            }
        )
        records.append(record)
    return {
        "summary": _summarize(records),
        "by_kind": {
            kind: _summarize(selected)
            for kind in GRAPH_GOLDEN_KINDS
            if (selected := [record for record in records if record["kind"] == kind])
        },
        "queries": records,
    }


def select_engines(engines: Sequence[str] | None) -> list[str]:
    """Resolve the ``--engine`` filter against the registry.

    Args:
        engines: Requested engine names, or ``None`` for every registered
            engine (the bake-off enumerates the registry; there is no
            ``GRAPH_ENGINE`` setting until stage 2).

    Returns:
        The engine names to run, sorted — the harness runs them sequentially
        (plan decision #15: one llama.cpp slot, and cognee's configuration is
        process-global).

    Raises:
        ValueError: If a requested name is not registered, or the filter
            selects nothing.
    """
    if engines is None:
        selected = sorted(GRAPH_ENGINE_REGISTRY)
    else:
        unknown = sorted({name for name in engines if name not in GRAPH_ENGINE_REGISTRY})
        if unknown:
            raise ValueError(
                f"unknown graph engine(s) {unknown}; registered: {sorted(GRAPH_ENGINE_REGISTRY)}"
            )
        selected = sorted(set(engines))
    if not selected:
        raise ValueError(f"no graph engines selected; registered: {sorted(GRAPH_ENGINE_REGISTRY)}")
    return selected


def engine_versions(names: Sequence[str]) -> dict[str, str | None]:
    """Stamp the installed version of each engine's library.

    Args:
        names: Engine names being run.

    Returns:
        Engine name → installed distribution version, or ``None`` when the
        library is not installed (which a run cannot actually reach, but a
        results file re-read later can).
    """
    versions: dict[str, str | None] = {}
    for name in names:
        distribution = ENGINE_DISTRIBUTIONS.get(name)
        if distribution is None:
            versions[name] = None
            continue
        try:
            versions[name] = version(distribution)
        except PackageNotFoundError:
            versions[name] = None
    return versions


@contextmanager
def open_engine_session(name: str, workdir: Path) -> Iterator[GraphSession]:
    """Open one registered engine's session, translating a missing library.

    Adapters import their engine libraries inside ``session()`` (plan decision
    #8), so a missing library surfaces here as a bare ``ModuleNotFoundError``
    from somewhere inside a third-party package. Translating it names the
    engine and the install command instead.

    Args:
        name: Registered engine name.
        workdir: The engine's working directory.

    Yields:
        The open session (closed on exit — teardown is not optional; Graphiti
        runs an embedded ``redis-server`` subprocess).

    Raises:
        RuntimeError: If the engine's library is not installed.
    """
    engine = get_graph_engine(name)
    try:
        with engine.session(workdir) as session:
            yield session
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"graph engine {name!r} needs a library that is not installed "
            f"(no module named {exc.name!r}). The bake-off engines live in the `bakeoff` "
            "dependency group — run `uv run --group bakeoff main.py eval graph` "
            "(or `uv sync --group bakeoff` first)."
        ) from exc


def _build_block(report: BuildReport, stats: GraphStats) -> dict[str, Any]:
    """Shape one build's report and the resulting graph size for the results.

    Args:
        report: The build's :class:`~varagity.graph.records.BuildReport`.
        stats: The graph's size after the build.

    Returns:
        The results-document block.
    """
    return {
        "messages_seen": report.messages_seen,
        "wall_clock_s": round(report.wall_clock_s, 2),
        "failures": list(report.failures),
        "stats": stats.model_dump(),
    }


def run_graph_eval(
    *,
    profile: str = "smoke",
    engines: Sequence[str] | None = None,
    mode: str | None = None,
    skip_build: bool = False,
    message_target: int | None = None,
    eval_root: Path = GRAPH_EVAL_ROOT,
    golden_path: Path = GRAPH_GOLDEN_PATH,
    results_dir: Path = RESULTS_DIR,
    seed: int = FIXTURE_SEED,
    verbose: int | None = None,
) -> dict[str, Any]:
    """Run the graph bake-off over the fixture corpus; persist results (§12).

    The sequence, once per selected engine (sequentially — plan decision #15):
    wipe its working directory, index the corpus, run the incremental
    re-index check, then ask every golden question and score the answers. The
    corpus is generated first and its manifest pins the parser's naming
    settings, so the messages the engines index say "Bob Nakamura" rather than
    a phone number.

    This harness decides ADR-017 and then stays on as the graph regression
    guard, exactly as the chat eval did for ADR-011.

    Args:
        profile: Fixture corpus profile — ``smoke`` (~226 scripted messages,
            the adapter/iteration corpus) or ``full`` (> 10,000 messages
            across a decade, the bake-off corpus).
        engines: Engine names to run; ``None`` runs every registered engine.
        mode: Engine query mode override for every question (``None`` uses
            each adapter's primary mode).
        skip_build: Reuse the existing corpus and each engine's already-built
            working directory, and skip the incremental check — the scoring
            iteration loop, which must not re-pay a multi-hour index.
        message_target: Cap the corpus at this many messages instead of the
            profile's own size — how an engine too slow for the uncapped
            corpus (Graphiti, ~51 s/message) still gets a full-profile seat.
            The scripted messages are always written, so the golden set holds;
            filler tops the corpus up to the cap. A capped run gets its own
            corpus and working directories (:func:`corpus_stem`).
        eval_root: Root for the generated corpus and the engines' working
            directories; resolved to an absolute path before use.
        golden_path: The graph golden Q&A file.
        results_dir: Directory for the timestamped results JSON.
        seed: Filler RNG seed for the generated corpus.
        verbose: Console verbosity (0–2); defaults to
            ``settings.DEFAULT_VERBOSE``.

    Returns:
        The results document (also written to ``results_dir``), with a
        ``"results_path"`` key naming the file.

    Raises:
        ValueError: If ``profile``, ``message_target`` or ``verbose`` is
            invalid, an engine name is not registered, or a golden provenance
            anchor is missing from the corpus.
        FileNotFoundError: If the golden set or a fixture input is missing.
        RuntimeError: If a selected engine's library is not installed.
    """
    if profile not in PROFILE_TARGETS:
        raise ValueError(
            f"unknown corpus profile {profile!r}; expected one of {[*PROFILE_TARGETS]}"
        )
    if message_target is not None and message_target < 1:
        raise ValueError(f"message_target must be a positive count, got {message_target}")
    verbose = check_verbose(get_settings().DEFAULT_VERBOSE if verbose is None else verbose)
    selected = select_engines(engines)

    # Engines consume their workdir verbatim, and cognee's config rejects a
    # relative path outright — so the CLI's repo-relative default must become
    # absolute before any session opens.
    eval_root = eval_root.resolve()
    corpus_dir = eval_root / "corpus"
    stem = corpus_stem(profile, message_target)
    db_path, manifest = prepare_corpus(
        corpus_dir,
        profile=profile,
        seed=seed,
        message_target=message_target,
        name=stem,
        reuse=skip_build,
    )
    if message_target is not None and manifest.message_count > message_target:
        logger.warning(
            "message_target %d is below the %d scripted message(s), which every corpus "
            "carries verbatim (the golden set is authored against them): the %s corpus "
            "holds %d",
            message_target,
            manifest.scripted_count,
            stem,
            manifest.message_count,
        )

    # The incremental re-index check (criterion §8.2#3) measures the marginal
    # cost of a re-export: the same messages plus a few new ones. Smoke only —
    # on the full profile the base index already costs hours — and never under
    # --skip-build, whose entire purpose is not to index.
    delta_manifest: FixtureManifest | None = None
    delta_path: Path | None = None
    if profile == "smoke" and not skip_build:
        delta_path, delta_manifest = prepare_corpus(
            corpus_dir,
            profile="full",
            seed=seed,
            message_target=manifest.message_count + INCREMENTAL_DELTA_MESSAGES,
            name=f"{stem}-delta",
        )

    corpora = [manifest] if delta_manifest is None else [manifest, delta_manifest]
    pins = manifest_settings_pins(*corpora)
    engine_results: dict[str, Any] = {}

    with pinned_graph_settings(**pins):
        batch = batch_for_path(db_path, corpus_dir, verbose=verbose)
        delta_batch: MessageBatch | None = (
            None if delta_path is None else batch_for_path(delta_path, corpus_dir, verbose=verbose)
        )
        entries = load_graph_golden(golden_path)
        validate_golden_against_manifest(entries, manifest)

        for name in selected:
            workdir = eval_root / stem / name
            if not skip_build:
                shutil.rmtree(workdir, ignore_errors=True)
            workdir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "graph eval: %s over the %s corpus (%d messages) in %s",
                name,
                stem,
                len(batch.messages),
                workdir,
            )
            started = time.monotonic()
            with open_engine_session(name, workdir) as session:
                build: dict[str, Any] | None = None
                if not skip_build:
                    report = session.build([batch], verbose=verbose)
                    build = _build_block(report, session.stats())
                    logger.info(
                        "%s indexed %d message(s) in %.1f s (%d failure(s)): %s",
                        name,
                        report.messages_seen,
                        report.wall_clock_s,
                        len(report.failures),
                        build["stats"],
                    )
                incremental: dict[str, Any] | None = None
                if delta_batch is not None:
                    before = session.stats()
                    delta_report = session.build([delta_batch], verbose=verbose)
                    incremental = _build_block(delta_report, session.stats())
                    incremental["new_messages"] = INCREMENTAL_DELTA_MESSAGES
                    incremental["stats_before"] = before.model_dump()
                    logger.info(
                        "%s re-indexed an overlapping batch (+%d new of %d) in %.1f s",
                        name,
                        INCREMENTAL_DELTA_MESSAGES,
                        delta_report.messages_seen,
                        delta_report.wall_clock_s,
                    )
                measured = measure_graph_engine(session, entries, mode=mode, verbose=verbose)
            engine_results[name] = {
                "workdir": str(workdir),
                "build": build,
                "incremental": incremental,
                "session_wall_clock_s": round(time.monotonic() - started, 2),
                **measured,
            }

    settings = get_settings()
    results: dict[str, Any] = {
        "kind": "graph_eval",
        "timestamp": datetime.now(UTC).isoformat(),
        "profile": profile,
        # The cap and the name it produced: a capped run's numbers are only
        # comparable to another run over the same corpus, and the stem is what
        # names both the corpus and every engine workdir recorded below.
        "message_target": message_target,
        "corpus_stem": stem,
        "seed": seed,
        "corpus_path": str(db_path),
        "golden_path": str(golden_path),
        "n_queries": len(entries),
        "kinds": list(GRAPH_GOLDEN_KINDS),
        "mode": mode,
        "skip_build": skip_build,
        "incremental_new_messages": None if delta_path is None else INCREMENTAL_DELTA_MESSAGES,
        "corpus": {
            "messages": manifest.message_count,
            "scripted": manifest.scripted_count,
            "filler": manifest.filler_count,
            "tapbacks": manifest.tapback_count,
            "threads": len(manifest.thread_counts),
            "first_timestamp": manifest.first_timestamp.isoformat(),
            "last_timestamp": manifest.last_timestamp.isoformat(),
        },
        "pinned_settings": pins,
        "chat_model": settings.BASE_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "engine_versions": engine_versions(selected),
        "engines": engine_results,
    }
    results["results_path"] = str(write_results("graph", results, results_dir=results_dir))
    return results
