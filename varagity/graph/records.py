"""Engine-independent graph records (spec_graphrag §5.2, §10.2).

Every bake-off adapter normalizes its engine's native payloads into these
models, so the harness scores three very different engines through one shape
and the ADR-017 tables compare like with like. They are pydantic (not
dataclasses) for the same reason :class:`~varagity.graph.sources.base.SourceMessage`
is: stage 2 puts them on the wire (graph evidence in the SSE stream, the
graph-export endpoint) and in persisted eval results.

Two deliberate honesty rules run through the models:

* **Optional counts mean "the engine cannot report this"**, not zero —
  :class:`GraphStats` fields are ``None`` when an engine exposes no way to
  ask (criterion §8.2#4 is *measured* honesty, not a hidden default).
* **:attr:`GraphEvidence.raw` keeps the engine-native payload** so the ADR
  autopsy can look at what an adapter *couldn't* map, rather than at a
  silently lossy projection.
"""

from typing import Any

from pydantic import BaseModel


class GraphEntity(BaseModel):
    """One node the engine surfaced as evidence for an answer.

    Attributes:
        name: The entity's canonical name, as the engine resolved it.
        type: The engine's entity type/category (``None`` when the engine
            does not type its nodes).
        summary: The engine's description of the entity (``None`` when it
            surfaces bare names).
    """

    name: str
    type: str | None = None
    summary: str | None = None


class GraphRelation(BaseModel):
    """One edge the engine surfaced as evidence for an answer.

    Attributes:
        source: Name of the edge's source entity.
        target: Name of the edge's target entity.
        label: Short relation label/keywords (``None`` when unlabelled).
        description: The engine's longer description of the relation — for
            Graphiti this is its *fact* string, which is the whole answer
            substrate (plan decision #12).
    """

    source: str
    target: str
    label: str | None = None
    description: str | None = None


class GraphCommunity(BaseModel):
    """One community (cluster) summary the engine surfaced.

    Only engines with a community tier ever populate this: Graphiti (label
    propagation + LLM summaries) does, LightRAG and cognee do not (R1).

    Attributes:
        id: The engine's community identifier.
        title: Human-facing community label (``None`` when unnamed).
        summary: The community's summary text.
    """

    id: str
    title: str | None = None
    summary: str


class GraphEvidence(BaseModel):
    """What the engine retrieved to support one answer (spec_graphrag §10.2).

    Attributes:
        entities: Nodes cited by the retrieval.
        relations: Edges cited by the retrieval.
        communities: Community summaries cited by the retrieval (empty for
            engines with no community layer).
        message_guids: Source-message provenance — ``[]`` when the engine
            surfaced nothing mappable back to a message. Empty is a
            *result*, not a failure: provenance recall is scored ``None``
            for such engines rather than 0 (criterion §8.2#4).
        raw: The engine-native payload the mapping was derived from, kept
            verbatim for the ADR autopsy (``None`` when the engine returned
            nothing structured).
    """

    entities: list[GraphEntity] = []
    relations: list[GraphRelation] = []
    communities: list[GraphCommunity] = []
    message_guids: list[str] = []
    raw: dict[str, Any] | None = None


class GraphAnswer(BaseModel):
    """One engine's answer to one question, with its evidence.

    Attributes:
        answer: The generated answer text, ``<think>``-stripped. For
            LightRAG and cognee this is the engine's own answer pipeline;
            for Graphiti it is our synthesis over its retrieved facts (plan
            decision #12 — the asymmetry the ADR records).
        evidence: What the engine retrieved to produce it.
        mode: The engine query mode actually used (the ``--mode`` escape
            hatch records extra passes under their own mode name).
        latency_s: Wall-clock seconds for the whole query, retrieval and
            generation together.
    """

    answer: str
    evidence: GraphEvidence
    mode: str
    latency_s: float


class BuildReport(BaseModel):
    """What one :meth:`~varagity.graph.base.GraphSession.build` call did.

    Attributes:
        messages_seen: Messages handed to the engine after guid-merging the
            batches (the upsert grain — a second build over an overlapping
            batch sees the union, not the sum).
        wall_clock_s: Seconds the indexing took. The headline bake-off
            number (§8.2#8) and the input to Phase 5's scheduling.
        failures: Human-readable failures the adapter caught and continued
            past (a rejected document, an extraction call that never
            returned). Empty means a clean build; the count is criterion
            §8.2#2 data, so failures are collected rather than raised.
    """

    messages_seen: int
    wall_clock_s: float
    failures: list[str] = []


class GraphStats(BaseModel):
    """Size of the graph an engine holds, as far as it will say.

    Every field is optional because "the engine exposes no count API" is a
    real and reportable state — the incremental-reindex check reads these
    before and after a delta build, and a ``None`` there is honest data for
    the ADR rather than a zero that reads like an empty graph.

    Attributes:
        entities: Entity/node count, or ``None`` if unavailable.
        relations: Relation/edge count, or ``None`` if unavailable.
        communities: Community count, or ``None`` if unavailable — which
            includes engines with no community layer at all.
    """

    entities: int | None = None
    relations: int | None = None
    communities: int | None = None
