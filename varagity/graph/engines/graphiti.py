"""The Graphiti bake-off adapter (spec_graphrag §8; ADR-017 candidate).

Graphiti is the episode-shaped candidate: one message becomes one timestamped
episode, which makes its provenance story the strongest of the three on paper
— an edge names the episodes it came from, and an episode's name *is* the
message guid. The bake-off is what tests that claim.

Three things make this adapter structurally different from the other two:

* **Storage is embedded FalkorDB Lite** — the ``falkordblite`` distribution
  ships a ``redislite`` module whose ``AsyncFalkorDB(<dbfile>)`` spawns a local
  ``redis-server`` subprocess (py3.12+ only, which this repo is), so teardown
  is not optional and the session context manager owns it:
  ``Graphiti.close()`` reaches the client's ``aclose()``, which shuts the
  subprocess down. No compose service is added in stage 1 (plan decision #9).
* **Its search returns facts, not prose.** ``search`` yields edge facts, so the
  answer is *our* synthesis over them: one grounded
  :meth:`~varagity.models.llm.LLMClient.generate` call plus
  :func:`~varagity.models.llm.clean_response` (plan decision #12). LightRAG and
  cognee are scored on their own answer pipelines, so this asymmetry is
  recorded in the ADR rather than papered over.
* **``SEMAPHORE_LIMIT`` defaults to 20 in source** (its docs say 10 — trust the
  source), which would fire twenty concurrent extraction calls at a
  single-slot llama.cpp. It is read at import time, so the pin lands before
  the lazy import.

Episodes are added under one corpus-wide ``group_id`` rather than one per
thread: an episode-shaped engine resolves and dedupes entities *within* a
partition, and the §12 Q1 questions are exactly the ones that need "Bob" in
three different threads to be one node.
"""

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from varagity.config import get_settings
from varagity.graph.base import GraphSession, register
from varagity.graph.records import (
    BuildReport,
    GraphAnswer,
    GraphCommunity,
    GraphEntity,
    GraphEvidence,
    GraphRelation,
    GraphStats,
)
from varagity.graph.render import episode_payloads, merge_batches
from varagity.graph.sources.base import MessageBatch
from varagity.models.llm import LLMClient, clean_response

logger = logging.getLogger(__name__)

# Graphiti has one retrieval entry point rather than named modes; the name is
# recorded on every answer so the results document reads uniformly across
# engines, and `--mode` still selects an alternative if one is ever added.
PRIMARY_MODE = "search"

# One partition for the whole corpus (see the module docstring).
GROUP_ID = "varagity"

# Read by graphiti_core at import time — 20 concurrent extraction calls would
# swamp a single-slot llama.cpp.
_ENV_PINS: dict[str, str] = {"SEMAPHORE_LIMIT": "1"}

# Generous cap for Graphiti's structured-output calls: OpenAIGenericClient
# injects the JSON schema into the prompt for local models, so replies are
# long, and its own default is 16384.
_LLM_MAX_TOKENS = 4096
# Cap for our synthesis call — an answer, not a document.
_SYNTHESIS_MAX_TOKENS = 1024

_EMBEDDING_DIM = 1024

SYNTHESIS_PROMPT = """\
You are answering a question about a personal message archive, using only the
facts retrieved from its knowledge graph.

Rules:
- Use ONLY the facts below. Do not invent people, events, dates, or opinions.
- If the facts do not answer the question, say so plainly.
- Answer in a few sentences, naming the people involved.

FACTS:
{facts}

QUESTION: {question}

ANSWER:"""


def facts_block(evidence: GraphEvidence) -> str:
    """Render retrieved relations and communities as the synthesis prompt's facts.

    Args:
        evidence: The normalized evidence from a Graphiti search.

    Returns:
        One ``- fact`` line per relation (community summaries appended after
        them), or ``""`` when the search found nothing — which is what makes
        the synthesis call skippable.
    """
    lines = [
        f"- {relation.description or relation.label or ''}".rstrip()
        for relation in evidence.relations
        if relation.description or relation.label
    ]
    lines.extend(
        f"- community {community.title or community.id}: {community.summary}"
        for community in evidence.communities
        if community.summary
    )
    return "\n".join(lines)


def synthesize(llm: LLMClient, question: str, evidence: GraphEvidence) -> str:
    """Turn retrieved facts into a grounded answer (plan decision #12).

    Args:
        llm: The chat client (:class:`~varagity.models.llm.LLMClient`, reused
            rather than a bespoke HTTP call, so the app's retry, clamp, and
            context-window discipline all apply).
        question: The question, verbatim.
        evidence: What the search retrieved.

    Returns:
        The ``<think>``-stripped answer, or a plain "no facts" sentence when
        the search returned nothing (the honest empty answer a fact-shaped
        engine produces, scored as-is).
    """
    facts = facts_block(evidence)
    if not facts:
        return "The graph returned no facts for this question."
    prompt = SYNTHESIS_PROMPT.format(facts=facts, question=question)
    try:
        raw = llm.generate(
            [{"role": "user", "content": prompt}], max_tokens=_SYNTHESIS_MAX_TOKENS, verbose=0
        )
    except Exception:  # a failed synthesis is a scored miss, not a dead run
        logger.warning("Graphiti answer synthesis failed", exc_info=True)
        return ""
    # Mandatory: generate() returns reasoning stages verbatim (the condense and
    # HyDE precedents), and an unstripped block would be scored as the answer.
    return clean_response(raw)


def evidence_from_search(results: Any, guid_by_uuid: Mapping[str, str]) -> GraphEvidence:
    """Normalize a Graphiti search result into engine-independent evidence.

    Handles both search surfaces: a bare list of edges (``search``) and the
    combined results object carrying ``edges``/``nodes``/``episodes``/
    ``communities`` (``search_``). Unknown shapes degrade to ``raw`` plus
    empty lists rather than raising.

    Args:
        results: The search results.
        guid_by_uuid: Episode uuid → message guid, recorded when the episodes
            were added, for the uuid-shaped provenance in edge payloads.

    Returns:
        The normalized evidence, including per-message provenance where the
        results named their episodes.
    """
    payload = _sections(results)
    entities = [entity for item in payload["nodes"] if (entity := _entity(item)) is not None]
    relations = [relation for item in payload["edges"] if (relation := _relation(item)) is not None]
    communities = [
        community for item in payload["communities"] if (community := _community(item)) is not None
    ]
    return GraphEvidence(
        entities=entities,
        relations=relations,
        communities=communities,
        message_guids=_message_guids(payload, guid_by_uuid),
        raw={key: value for key, value in payload.items() if value},
    )


@register("graphiti")
class GraphitiEngine:
    """Graphiti behind the :class:`~varagity.graph.base.GraphEngine` protocol."""

    @contextmanager
    def session(self, workdir: Path) -> Iterator[GraphSession]:
        """Open a Graphiti session storing everything under ``workdir``.

        Args:
            workdir: The engine's working directory (created if absent).

        Yields:
            The open session; the embedded FalkorDB Lite subprocess and the
            session's event loop are both torn down on exit.
        """
        settings = get_settings()
        workdir.mkdir(parents=True, exist_ok=True)
        _pin_env()
        # Lazy, so the registry costs nothing without the bakeoff group
        # (plan decision #8) and so SEMAPHORE_LIMIT above is already in place.
        # redislite is the falkordblite distribution's import name; its
        # AsyncFalkorDB duck-types falkordb.asyncio.FalkorDB, which is the
        # shape FalkorDriver(falkor_db=…) drives.
        from graphiti_core import Graphiti
        from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
        from graphiti_core.driver.falkordb_driver import FalkorDriver
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
        from graphiti_core.nodes import EpisodeType
        from redislite import AsyncFalkorDB

        database = AsyncFalkorDB(str(workdir / "falkordb.db"))
        llm_config = LLMConfig(
            api_key=settings.BASE_MODEL_API_KEY,
            base_url=settings.BASE_MODEL_API_URL,
            model=settings.BASE_MODEL,
            max_tokens=_LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
        )
        graphiti = Graphiti(
            graph_driver=FalkorDriver(falkor_db=database),
            llm_client=OpenAIGenericClient(config=llm_config),
            embedder=OpenAIEmbedder(
                config=OpenAIEmbedderConfig(
                    api_key=settings.EMBEDDING_API_KEY,
                    base_url=settings.EMBEDDING_API_URL,
                    embedding_model=settings.EMBEDDING_MODEL,
                    embedding_dim=_EMBEDDING_DIM,
                )
            ),
            # Without an explicit cross-encoder Graphiti constructs its
            # default OpenAI reranker, which requires a real OPENAI_API_KEY
            # at construction time (verified live, Phase 3 gate). The plain
            # `search()` path this adapter uses never ranks with it; pointing
            # it at llama.cpp keeps construction local either way.
            cross_encoder=OpenAIRerankerClient(config=llm_config),
        )
        session = _GraphitiSession(graphiti, EpisodeType.message, llm=LLMClient())
        try:
            session.run(graphiti.build_indices_and_constraints())
            yield session
        finally:
            session.close()


class _GraphitiSession:
    """One open Graphiti working session (see :class:`GraphitiEngine`)."""

    def __init__(
        self,
        graphiti: Any,
        episode_type: Any,
        *,
        llm: LLMClient,
        mode: str = PRIMARY_MODE,
    ) -> None:
        """Wrap an initialized Graphiti instance.

        Args:
            graphiti: The ``Graphiti`` instance (injected so the session's
                logic is testable against a double, with only
                :meth:`GraphitiEngine.session` touching the real library).
            episode_type: ``EpisodeType.message``, the source kind for chat.
            llm: Chat client for the answer synthesis (plan decision #12).
            mode: Recorded query mode name.
        """
        self._graphiti = graphiti
        self._episode_type = episode_type
        self._llm = llm
        self._mode = mode
        self._guid_by_uuid: dict[str, str] = {}
        self._added: set[str] = set()
        self._loop = asyncio.new_event_loop()

    def run[T](self, awaitable: Awaitable[T]) -> T:
        """Run one coroutine on the session's event loop.

        Args:
            awaitable: The coroutine to drive.

        Returns:
            Whatever it returned.
        """
        return self._loop.run_until_complete(awaitable)

    def build(self, batches: Sequence[MessageBatch], *, verbose: int = 0) -> BuildReport:
        """Add one episode per merged message, then rebuild communities.

        Episode identity is the message guid, and guids already added in this
        session are skipped — which is how a second build over an overlapping
        batch stays an upsert rather than a duplication.

        Args:
            batches: Parsed source files (guid-merged before rendering).
            verbose: Validated console verbosity (0–2).

        Returns:
            The build report; per-episode failures are recorded and the run
            continues (criterion §8.2#2 counts them).
        """
        messages = merge_batches(batches)
        payloads = [
            payload
            for payload in episode_payloads(messages, group_id=GROUP_ID)
            if payload.name not in self._added
        ]
        failures: list[str] = []
        started = time.perf_counter()
        for payload in payloads:
            try:
                result = self.run(
                    self._graphiti.add_episode(
                        name=payload.name,
                        episode_body=payload.body,
                        source=self._episode_type,
                        source_description=payload.source_description,
                        reference_time=payload.reference_time,
                        group_id=payload.group_id,
                        # Never update_communities=True: 0.29.2's per-episode
                        # community path unpacks a variable-length gather and
                        # fails every episode touching ≠2 community nodes
                        # (verified live, Phase 3 gate). The build_communities()
                        # pass below is the working surface.
                    )
                )
            except Exception as exc:  # one bad episode must not end the index
                logger.warning("Graphiti add_episode failed for %s", payload.name, exc_info=True)
                failures.append(f"episode {payload.name}: {exc!r}")
                continue
            self._added.add(payload.name)
            uuid = _episode_uuid(result)
            if uuid is not None:
                self._guid_by_uuid[uuid] = payload.name
        if payloads:
            try:
                self.run(self._graphiti.build_communities())
            except Exception as exc:
                logger.warning("Graphiti build_communities failed", exc_info=True)
                failures.append(f"build_communities: {exc!r}")
        return BuildReport(
            messages_seen=len(messages),
            wall_clock_s=time.perf_counter() - started,
            failures=failures,
        )

    def query(self, question: str, *, mode: str | None = None, verbose: int = 0) -> GraphAnswer:
        """Retrieve facts from the graph and synthesize an answer over them.

        Args:
            question: The question, verbatim.
            mode: Recorded mode name; ``None`` uses the session's primary.
            verbose: Validated console verbosity (0–2).

        Returns:
            The synthesized answer with its normalized evidence.
        """
        used = mode or self._mode
        started = time.perf_counter()
        evidence = evidence_from_search(self._search(question), self._guid_by_uuid)
        return GraphAnswer(
            answer=synthesize(self._llm, question, evidence),
            evidence=evidence,
            mode=used,
            latency_s=time.perf_counter() - started,
        )

    def stats(self) -> GraphStats:
        """Count entity, relation, and community records through the driver.

        Returns:
            The three counts, each ``None`` when the query would not run.
            Graphiti is the one candidate that can answer the community
            question at all (label propagation + LLM summaries).
        """
        return GraphStats(
            entities=self._count("MATCH (n:Entity) RETURN count(n) AS value"),
            relations=self._count("MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS value"),
            communities=self._count("MATCH (n:Community) RETURN count(n) AS value"),
        )

    def close(self) -> None:
        """Close Graphiti (and its embedded database) and the session's loop."""
        try:
            self.run(self._graphiti.close())
        except Exception:
            logger.warning("Graphiti teardown failed", exc_info=True)
        finally:
            self._loop.close()

    def _search(self, question: str) -> Any:
        """Retrieve facts for a question, tolerating retrieval failure.

        Args:
            question: The question, verbatim.

        Returns:
            The search results, or ``None`` when the search could not run
            (the turn is then a scored miss, not a dead run).
        """
        try:
            return self.run(self._graphiti.search(question))
        except Exception:
            logger.warning("Graphiti search failed", exc_info=True)
            return None

    def _count(self, query: str) -> int | None:
        """Run one counting Cypher query through Graphiti's driver.

        Args:
            query: A query returning a single ``value`` column.

        Returns:
            The count, or ``None`` when the driver would not answer.
        """
        try:
            records = self.run(self._graphiti.driver.execute_query(query))
        except Exception:
            logger.warning("Graphiti stats query failed: %s", query, exc_info=True)
            return None
        return _first_count(records)


def _pin_env() -> None:
    """Pin graphiti_core's import-time environment knobs to single-slot values."""
    os.environ.update(_ENV_PINS)


def _sections(results: Any) -> dict[str, list[Mapping[str, Any]]]:
    """Split a search result into its edge/node/episode/community records.

    Args:
        results: A results object (``search_``), a list of edges (``search``),
            or anything else.

    Returns:
        A mapping with the four keys, each holding plain-data records (empty
        lists for whatever the result did not carry).
    """
    sections: dict[str, list[Mapping[str, Any]]] = {
        "edges": [],
        "nodes": [],
        "episodes": [],
        "communities": [],
    }
    if results is None:
        return sections
    if isinstance(results, Sequence) and not isinstance(results, str):
        sections["edges"] = _records(results)
        return sections
    return {name: _records(getattr(results, name, None)) for name in sections}


def _records(value: Any) -> list[Mapping[str, Any]]:
    """Coerce a sequence of Graphiti models into plain mappings.

    Args:
        value: A sequence of pydantic models or mappings, or ``None``.

    Returns:
        The mappings, skipping anything that is neither.
    """
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    records: list[Mapping[str, Any]] = []
    for item in value:
        dump = getattr(item, "model_dump", None)
        record = dump(mode="json") if callable(dump) else item
        if isinstance(record, Mapping):
            records.append(record)
    return records


def _message_guids(
    payload: Mapping[str, list[Mapping[str, Any]]], guid_by_uuid: Mapping[str, str]
) -> list[str]:
    """Recover message guids from a search result's episodes and edges.

    Episode names *are* message guids, so a result carrying episodes is
    directly provenanced; edges only carry episode uuids, which resolve
    through the map built while the episodes were added (and therefore only
    within a session that did the build — a ``--skip-build`` re-score reports
    no edge-side provenance, honestly).

    Args:
        payload: The split search result.
        guid_by_uuid: Episode uuid → message guid.

    Returns:
        The cited message guids, deduplicated, episodes first.
    """
    guids: list[str] = []
    seen: set[str] = set()

    def add(guid: str | None) -> None:
        if guid and guid not in seen:
            seen.add(guid)
            guids.append(guid)

    for episode in payload["episodes"]:
        add(_text(episode, "name"))
    for edge in payload["edges"]:
        uuids = edge.get("episodes")
        if isinstance(uuids, Sequence) and not isinstance(uuids, str):
            for uuid in uuids:
                add(guid_by_uuid.get(uuid) if isinstance(uuid, str) else None)
    return guids


def _episode_uuid(result: Any) -> str | None:
    """Read the episode uuid out of an ``add_episode`` result.

    Args:
        result: Whatever ``add_episode`` returned.

    Returns:
        The uuid, or ``None`` when the result did not carry one.
    """
    episode = getattr(result, "episode", None)
    uuid = getattr(episode, "uuid", None)
    return uuid if isinstance(uuid, str) else None


def _first_count(records: Any) -> int | None:
    """Read the first integer out of a driver's counting-query result.

    Drivers differ in shape (a list of records, a ``(records, …)`` tuple, a
    list of dicts), so the read is structural: the first integer found wins.

    Args:
        records: The driver's return value.

    Returns:
        The count, or ``None`` when no integer was found.
    """
    if isinstance(records, bool):  # bool is an int subclass — never a count
        return None
    if isinstance(records, int):
        return records
    if isinstance(records, Mapping):
        return _first_count(list(records.values()))
    if isinstance(records, Sequence) and not isinstance(records, str | bytes):
        for item in records:
            found = _first_count(item)
            if found is not None:
                return found
    return None


def _text(item: Mapping[str, Any], *names: str) -> str | None:
    """Read the first present non-empty string field from a record.

    Args:
        item: One node, edge, episode, or community record.
        *names: Candidate field names, most likely first.

    Returns:
        The trimmed value, or ``None`` when no candidate holds text.
    """
    for name in names:
        value = item.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _entity(item: Mapping[str, Any]) -> GraphEntity | None:
    """Map one Graphiti entity node.

    Args:
        item: The record.

    Returns:
        The entity, or ``None`` when it carries no usable name.
    """
    name = _text(item, "name", "uuid")
    if name is None:
        return None
    return GraphEntity(
        name=name,
        type=_specific_label(item) or _text(item, "entity_type", "type"),
        summary=_text(item, "summary", "description"),
    )


def _specific_label(item: Mapping[str, Any]) -> str | None:
    """Read a node's most specific graph label.

    Every entity node carries the generic ``Entity`` label plus whatever the
    extraction assigned, so the first non-generic label is the type.

    Args:
        item: The record.

    Returns:
        The label, or ``None`` when the node has only the generic one.
    """
    labels = item.get("labels")
    if not isinstance(labels, Sequence) or isinstance(labels, str):
        return None
    return next((label for label in labels if isinstance(label, str) and label != "Entity"), None)


def _relation(item: Mapping[str, Any]) -> GraphRelation | None:
    """Map one Graphiti entity edge.

    Graphiti's edges carry their meaning in ``fact`` — a sentence, not a
    label — which is exactly the substrate the synthesis call answers from.

    Args:
        item: The record.

    Returns:
        The relation, or ``None`` when both endpoints and the fact are
        missing.
    """
    source = _text(item, "source_node_uuid", "source", "src")
    target = _text(item, "target_node_uuid", "target", "tgt")
    fact = _text(item, "fact", "description", "summary")
    if source is None and target is None and fact is None:
        return None
    return GraphRelation(
        source=source or "",
        target=target or "",
        label=_text(item, "name", "label", "relation"),
        description=fact,
    )


def _community(item: Mapping[str, Any]) -> GraphCommunity | None:
    """Map one Graphiti community node.

    Args:
        item: The record.

    Returns:
        The community, or ``None`` when it carries no summary.
    """
    summary = _text(item, "summary", "description")
    if summary is None:
        return None
    return GraphCommunity(
        id=_text(item, "uuid", "id") or "",
        title=_text(item, "name", "title"),
        summary=summary,
    )
