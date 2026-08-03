"""The cognee bake-off adapter (spec_graphrag §8; ADR-017 candidate).

cognee is the other document-shaped candidate, and the one whose stack fit is
strongest for stage 2: its graph adapter for Postgres is **plain SQL** (no
Apache AGE extension, unlike LightRAG's), so a winning cognee could move into
the pgvector database this repo already runs. The bake-off itself uses its
shipped embedded defaults — sqlite + lancedb + ladybug, all inside the
session's working directory — because engines self-store in stage 1 (plan
decision #9) and production storage is ADR-017's decision, not this phase's.

Configuration is environment-driven and read at import time
(``EMBEDDING_PROVIDER="openai_compatible"`` is the direct-SDK path that skips
LiteLLM and takes a 1024-dim local endpoint; ``LLM_PROVIDER="custom"`` is the
documented self-hosted OpenAI-compatible path), so the pins land *before* the
lazy import — the same mechanism, and the same caveat, as the LightRAG adapter.

**The ``<think>`` trap has no clean injection point here.** cognee drives
structured output through LiteLLM + instructor, which parses the completion
into pydantic models inside the library; there is no callable to wrap the way
LightRAG's ``llm_model_func`` can be wrapped. instructor's own validation
retries are the mitigation, and the resulting failure count is bake-off data
(criterion §8.2#2) rather than something this adapter hides.

**Two knobs exist only because llama.cpp serves one request at a time.**
cognee's extraction task fans an *unbounded* ``asyncio.gather`` out over every
chunk of a pipeline batch, with summarization gathering alongside it, so the
batch size *is* the concurrency: at cognee's default of 100 that is ~200
completions in flight against a ``--parallel 1`` server, where the tail
request's queue wait alone outlives LiteLLM's fallback deadline — instructor
then retries it, and the cascade takes the whole build down (observed
2026-07-28 on the first full-profile run: 76 min, zero entities).
:data:`_CHUNKS_PER_BATCH` bounds the fan-out — free here, since the requests
serialize either way — and ``REQUEST_TIMEOUT`` raises the deadline.

That variable is read by LiteLLM at *its* import time and is deliberately
process-wide: it governs every LiteLLM completion in the process, the
**search** path's answer generation included (queries run through the same
``GenericAPIAdapter``, which passes no per-call timeout either). Its generic
name would be a leak worth worrying about in a long-lived process; the
bake-off's sessions are process-scoped and its harness already pins the
environment per session, so that reach is exactly the intended one.

**The second full-profile attempt died differently, and that is why ingestion
is grouped.** With the fan-out bounded, extraction stopped timing out and
started *spiralling*: on content-heavy chunks the reasoning model emitted
reasoning until the served window was full and then returned ``content=''``
with ``finish_reason='length'``. The log's own arithmetic is exact
(2026-07-29): prompt 4,805 + completion 11,579 = a full 16,384-token window,
and the retry that followed burned 23,318 completion tokens against a 32,768
one — twice the headroom bought nothing but a slower death, because nothing
in cognee caps a completion's length (its ``max_completion_tokens`` reaches
only the transcription calls). instructor then makes each attempt worse than
the last: its retry prompt embeds the failed ``ModelResponse`` verbatim, which
took that same call from 4,805 to 9,450 prompt tokens. When the third attempt
is exhausted the ``RetryError`` leaves the data item, cognee raises
``PipelineRunFailedError`` for the whole run, and ``cognify_rollback_handler``
deletes every node and edge carrying that ``pipeline_run_id`` — which is how
one poisoned chunk turned a 3.1-hour build into zero entities with no partial
progress.

:data:`_DOCS_PER_COGNIFY` is the containment. Transcripts are added and
cognified in groups, so one ``cognify`` call — and therefore one rollback — is
the blast radius, and every finished group is durable before the next starts.
The dataset stays **single** throughout, because search must span the whole
corpus; grouping it this way is cognee's own incremental pattern rather than a
trick, since ``cognify(incremental_loading=True)`` (its default) skips any
data item already marked completed for this pipeline and dataset, so each
group's pass over the earlier documents costs one status read apiece and no
LLM call.
"""

import asyncio
import logging
import os
import re
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
    GraphEntity,
    GraphEvidence,
    GraphRelation,
    GraphStats,
)
from varagity.graph.render import (
    TranscriptDoc,
    doc_guid_index,
    guids_in_payload,
    merge_batches,
    thread_transcripts,
)
from varagity.graph.sources.base import MessageBatch

logger = logging.getLogger(__name__)

# cognee's default search type and the adapter's primary (17 exist on 1.4;
# this one retrieves over the graph and generates an answer). `CHUNKS` is the
# evidence pass: retrieval-only, and its records carry `document_name` — the
# thread back to message provenance. The non-generative graph-record search
# the plan assumed (0.x `INSIGHTS`) no longer exists on 1.4 (verified live,
# Phase 3 gate), so question-scoped entities/relations are not surfaceable
# through cognee's search API — itself criterion §8.2#4 data.
PRIMARY_MODE = "GRAPH_COMPLETION"
_EVIDENCE_MODE = "CHUNKS"

# One dataset per session — cognee scopes ingestion and search by dataset, and
# the harness's working directories are already per-engine and per-profile.
DATASET = "varagity_graph"

# Transcripts are handed to cognee as files so their names ride into its
# metadata: that file stem is the only thread the provenance walk has back
# from a returned chunk to the messages behind it.
_DOCUMENTS_DIR = "documents"
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")

_EMBEDDING_DIM = 1024

# Chunks per cognify batch (cognee's own default is 100). The batch is the
# extraction task's `asyncio.gather` width, and summarization gathers over it
# in parallel, so this bounds roughly 2N concurrent completions. A small bound
# costs nothing on a single-slot llama.cpp — the requests queue either way —
# and it is what keeps the tail request's queue wait structurally inside
# `_REQUEST_TIMEOUT_S` instead of merely usually inside it.
_CHUNKS_PER_BATCH = 8

# Transcript files per add/cognify pass. Each `cognify()` is one cognee
# pipeline run, and a failed run is rolled back *whole* — every node and edge
# carrying its `pipeline_run_id` is deleted and the affected documents'
# completion status is cleared (`cognify_rollback_handler`, verified in the
# installed package and observed live). The group size is therefore the
# failure blast radius, which is the entire point of grouping: one poisoned
# chunk costs one group, not the build. What a smaller group costs is one
# pipeline startup (migration check, task construction, dataset resolution,
# plus a status read per already-completed document) times the number of
# groups; at 100 that overhead disappears next to a group's extraction cost.
_DOCS_PER_COGNIFY = 100

# `cognify(chunk_size=…)` is deliberately left at cognee's default, which
# resolves to `min(EMBEDDING_MAX_COMPLETION_TOKENS, LLM_MAX_COMPLETION_TOKENS
# // 2)` = min(8191, 8192) = 8191 tokens. That ceiling never binds here:
# `thread_transcripts` already caps a document at 8000 characters and
# `TextChunker` packs whole paragraphs up to the ceiling, so one transcript is
# one chunk and *our* cap is what sizes the extraction prompt (measured at
# 4,805 tokens, roughly two-thirds transcript and one-third schema plus
# instructions). Pinning something smaller would only sub-split documents this
# adapter already sized: it would buy ~2k tokens of think-headroom against a
# spiral that ate 23,318 of them, multiply the extraction calls — and so a
# single-slot build's wall-clock — by the split factor, and shift extraction
# granularity, which is one of the things the bake-off is measuring. Isolation
# is the fix; the granularity stays cognee's own.

# LiteLLM's request deadline, in seconds. Left alone it resolves to 600 s for
# chat completions (`litellm.request_timeout` holds a 6000 s sentinel that
# `CompletionTimeout.resolve` maps down to `COMPLETION_HTTP_FALLBACK_SECONDS`
# whenever nothing configured it), and cognee's adapter passes no per-call
# timeout — so a queued request that waits longer than that dies, repeatedly.
# The value is read from the environment by `litellm.constants` at import
# time, which is why it is pinned rather than passed.
_REQUEST_TIMEOUT_S = 3600

# One INSIGHTS result: the (source node, edge, target node) mappings.
_Triplet = tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]


def transcript_filename(doc_key: str) -> str:
    """Make a filesystem-safe file name that still carries the document key.

    Thread guids contain characters (``;``, ``+``, ``:``) that a file name
    cannot, so they are collapsed to underscores. The stem is indexed
    alongside the raw key, which is what lets
    :func:`~varagity.graph.render.guids_in_payload` recognize a returned
    ``/…/documents/<stem>.txt`` path as a citation.

    Args:
        doc_key: The transcript's ``{thread_id}::{day-span}`` key.

    Returns:
        The file name, including its ``.txt`` suffix.
    """
    return f"{_UNSAFE_NAME.sub('_', doc_key).strip('_')}.txt"


def write_transcripts(docs: Sequence[TranscriptDoc], directory: Path) -> list[Path]:
    """Write transcripts into ``directory``, one file per document.

    Rewriting a document with unchanged content is what makes cognee's
    ingestion idempotent: ``data.id`` is a content hash, so an unchanged file
    is skipped and a changed one updates in place.

    Args:
        docs: Rendered transcripts.
        directory: Destination directory (created if absent).

    Returns:
        The written paths, in document order.
    """
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for doc in docs:
        path = directory / transcript_filename(doc.doc_key)
        path.write_text(doc.text, encoding="utf-8")
        paths.append(path)
    return paths


def provenance_index(docs: Sequence[TranscriptDoc]) -> dict[str, list[str]]:
    """Index transcripts under both their key and their on-disk file stem.

    Args:
        docs: Rendered transcripts.

    Returns:
        ``doc_key`` and file stem → the document's message guids.
    """
    index = doc_guid_index(docs)
    for doc in docs:
        index[Path(transcript_filename(doc.doc_key)).stem] = list(doc.message_guids)
    return index


def answer_from_results(results: Any) -> str:
    """Flatten cognee's search results into one answer string.

    ``search`` returns a list whose element type depends on the search type —
    strings for the completion types, richer objects otherwise. Everything
    string-shaped is joined; anything else is stringified rather than dropped,
    so a shape change shows up as an odd answer instead of an empty one.

    Args:
        results: Whatever ``cognee.search`` returned.

    Returns:
        The answer text (``""`` when there was nothing to say).
    """
    if results is None:
        return ""
    if isinstance(results, str):
        return results.strip()
    if isinstance(results, Sequence):
        return "\n".join(
            part for item in _result_items(results) if (part := str(item).strip())
        ).strip()
    return str(results).strip()


def _result_items(results: Sequence[Any]) -> Iterator[Any]:
    """Flatten cognee's per-dataset search-result wrappers.

    ``search`` returns either plain results or one mapping per dataset whose
    actual results sit under ``search_result`` (the wrapped shape observed
    live on 1.4 — Phase 3 gate). Both flatten to the same stream; anything
    else passes through untouched.

    Args:
        results: The ``search`` return value, already known to be a sequence.

    Yields:
        Each result item, unwrapped.
    """
    for item in results:
        if isinstance(item, Mapping):
            inner = item.get("search_result")
            if isinstance(inner, Sequence) and not isinstance(inner, str):
                yield from inner
                continue
        yield item


def evidence_from_search(
    retrieved: Any, completion: Any, index: Mapping[str, Sequence[str]]
) -> GraphEvidence:
    """Normalize a cognee evidence-pass result into engine-independent evidence.

    The evidence pass is ``CHUNKS`` (retrieval-only): its records carry
    ``document_name``, which the provenance walk resolves back to message
    guids. Triplet-shaped payloads — ``(source node, edge, target node)`` —
    still map to entities/relations for any mode that returns them; unknown
    shapes degrade to ``raw`` plus empty lists rather than raising.

    Args:
        retrieved: The evidence-pass search results.
        completion: The primary-mode results, walked for provenance alongside
            the retrieved records.
        index: Document key/stem → message guids.

    Returns:
        The normalized evidence. cognee has no community layer (R1), so
        ``communities`` is always empty.
    """
    entities: list[GraphEntity] = []
    relations: list[GraphRelation] = []
    seen: set[str] = set()
    for source, edge, target in _triplets(retrieved):
        source_name = _text(source, "name", "id", "text")
        target_name = _text(target, "name", "id", "text")
        for node, name in ((source, source_name), (target, target_name)):
            if name is not None and name not in seen:
                seen.add(name)
                entities.append(
                    GraphEntity(
                        name=name,
                        type=_text(node, "type", "node_type", "label"),
                        summary=_text(node, "description", "summary", "text"),
                    )
                )
        if source_name is not None and target_name is not None:
            relations.append(
                GraphRelation(
                    source=source_name,
                    target=target_name,
                    label=_text(edge, "relationship_name", "relationship_type", "name", "label"),
                    description=_text(edge, "description", "summary"),
                )
            )
    return GraphEvidence(
        entities=entities,
        relations=relations,
        communities=[],
        message_guids=guids_in_payload([retrieved, completion], index),
        raw={"evidence_pass": _jsonable(retrieved)},
    )


@register("cognee")
class CogneeEngine:
    """cognee behind the :class:`~varagity.graph.base.GraphEngine` protocol."""

    @contextmanager
    def session(self, workdir: Path) -> Iterator[GraphSession]:
        """Open a cognee session storing everything under ``workdir``.

        Args:
            workdir: The engine's working directory (created if absent).

        Yields:
            The open session.
        """
        workdir.mkdir(parents=True, exist_ok=True)
        _pin_env(workdir)
        # Lazy, so the registry costs nothing without the bakeoff group
        # (plan decision #8); cognee's configuration is process-global, which
        # is also why the harness runs engines sequentially (decision #15).
        import cognee
        from cognee.infrastructure.databases.graph import get_graph_engine
        from cognee.modules.search.types import SearchType

        cognee.config.data_root_directory(str(workdir / "data"))
        cognee.config.system_root_directory(str(workdir / "system"))
        session = _CogneeSession(cognee, SearchType, workdir=workdir, graph_engine=get_graph_engine)
        try:
            yield session
        finally:
            session.close()


class _CogneeSession:
    """One open cognee working session (see :class:`CogneeEngine`)."""

    def __init__(
        self,
        api: Any,
        search_types: Any,
        *,
        workdir: Path,
        graph_engine: Any,
        mode: str = PRIMARY_MODE,
    ) -> None:
        """Wrap the configured cognee module.

        Args:
            api: The ``cognee`` module (injected so the session's logic is
                testable against a double, with only :meth:`CogneeEngine.session`
                touching the real library).
            search_types: cognee's ``SearchType`` enum.
            workdir: The session's working directory.
            graph_engine: cognee's ``get_graph_engine`` coroutine factory, for
                :meth:`stats`.
            mode: Primary search type name for this session.
        """
        self._api = api
        self._search_types = search_types
        self._workdir = workdir
        self._graph_engine = graph_engine
        self._mode = mode
        self._index: dict[str, list[str]] = {}
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
        """Ingest the batches' merged messages as transcript files, then cognify.

        Args:
            batches: Parsed source files (guid-merged before rendering).
            verbose: Validated console verbosity (0–2).

        Returns:
            The build report. ``add``/``cognify`` failures are recorded per
            group rather than raised — a partial graph is still scoreable, and
            the failure count is criterion §8.2#2 data.
        """
        messages = merge_batches(batches)
        docs = thread_transcripts(messages)
        self._index.update(provenance_index(docs))
        paths = write_transcripts(docs, self._workdir / _DOCUMENTS_DIR)
        started = time.perf_counter()
        failures = self._ingest(paths) if paths else []
        return BuildReport(
            messages_seen=len(messages),
            wall_clock_s=time.perf_counter() - started,
            failures=failures,
        )

    def _ingest(self, paths: Sequence[Path]) -> list[str]:
        """Run cognee's two ingestion stages over the transcripts, group by group.

        Every group is one ``add`` plus one ``cognify``, and a ``cognify`` is
        one cognee pipeline run — the unit its rollback works in. A group that
        fails is therefore rolled back alone and the loop moves on to the next
        one, which is what keeps a single poisoned chunk from zeroing a
        multi-hour build (see the module docstring). The dataset is the same
        for every group: cognee's incremental loading skips the documents it
        has already completed, so search still spans everything while the
        widening dataset costs a status read per document instead of an LLM
        call.

        ``cognify`` is bounded to :data:`_CHUNKS_PER_BATCH` chunks per batch:
        its extraction stage gathers over a whole batch at once, which on a
        single-slot server turns the batch size into a queue depth.

        Args:
            paths: The written transcript files, in document order.

        Returns:
            Human-readable failures (empty on a clean ingest), one per failed
            stage and tagged with the group's position — ``"cognify[3/21]:
            RetryError(…)"``. A group whose ``add`` failed is not cognified
            (there would be nothing new to build a graph from), but every
            later group still runs.
        """
        groups = _document_groups(paths)
        failures: list[str] = []
        for number, group in enumerate(groups, start=1):
            where = f"[{number}/{len(groups)}]"
            try:
                self.run(self._api.add([str(path) for path in group], dataset_name=DATASET))
            except Exception as exc:  # a failed group is data, not the end of the run
                logger.warning("cognee add failed for group %s", where, exc_info=True)
                failures.append(f"add{where}: {exc!r}")
                continue
            try:
                self.run(self._api.cognify(datasets=[DATASET], chunks_per_batch=_CHUNKS_PER_BATCH))
            except Exception as exc:  # likewise: the next group is still worth ingesting
                logger.warning("cognee cognify failed for group %s", where, exc_info=True)
                failures.append(f"cognify{where}: {exc!r}")
        return failures

    def query(self, question: str, *, mode: str | None = None, verbose: int = 0) -> GraphAnswer:
        """Answer one question with cognee's own graph-completion pipeline.

        A second, generation-free ``CHUNKS`` search supplies the evidence —
        the same shape as the LightRAG adapter's context pass.

        Args:
            question: The question, verbatim.
            mode: A ``SearchType`` member name; ``None`` uses the session's
                primary.
            verbose: Validated console verbosity (0–2).

        Returns:
            The answer with its normalized evidence.
        """
        used = mode or self._mode
        started = time.perf_counter()
        completion = self._search(question, used)
        retrieved = self._search(question, _EVIDENCE_MODE)
        return GraphAnswer(
            answer=answer_from_results(completion),
            evidence=evidence_from_search(retrieved, completion, self._index),
            mode=used,
            latency_s=time.perf_counter() - started,
        )

    def stats(self) -> GraphStats:
        """Count nodes and edges through cognee's graph engine.

        Returns:
            Entity/relation counts, or ``None`` for each if the graph engine
            would not say. ``communities`` is always ``None``: cognee has no
            community-detection tier (R1).
        """
        entities = relations = None
        try:
            engine = self.run(self._graph_engine())
            nodes, edges = self.run(engine.get_graph_data())
            entities, relations = len(nodes), len(edges)
        except Exception:  # an engine that won't report is reported as unknown
            logger.warning("cognee graph stats unavailable", exc_info=True)
        return GraphStats(entities=entities, relations=relations, communities=None)

    def close(self) -> None:
        """Close the session's event loop."""
        self._loop.close()

    def _search(self, question: str, mode: str) -> Any:
        """Run one cognee search, tolerating an unknown mode or a failed call.

        Args:
            question: The question, verbatim.
            mode: A ``SearchType`` member name.

        Returns:
            The search results, or ``None`` when the search could not run.
        """
        query_type = getattr(self._search_types, mode, None)
        if query_type is None:
            logger.warning("unknown cognee search type %r — skipping that pass", mode)
            return None
        try:
            return self.run(
                self._api.search(query_text=question, query_type=query_type, datasets=[DATASET])
            )
        except Exception:
            logger.warning("cognee %s search failed", mode, exc_info=True)
            return None


def _document_groups(paths: Sequence[Path]) -> list[Sequence[Path]]:
    """Split the transcripts into :data:`_DOCS_PER_COGNIFY`-sized groups.

    Args:
        paths: The written transcript files, in document order.

    Returns:
        The groups in order, the last one holding the remainder; empty for an
        empty corpus.
    """
    return [
        paths[start : start + _DOCS_PER_COGNIFY]
        for start in range(0, len(paths), _DOCS_PER_COGNIFY)
    ]


def _pin_env(workdir: Path) -> None:
    """Pin cognee's import-time environment to the local endpoints and workdir.

    Args:
        workdir: The session's working directory (the embedded stores' home).
    """
    os.environ.update(_env_pins(workdir))


def _env_pins(workdir: Path) -> dict[str, str]:
    """Build cognee's import-time environment (kept separate so it is testable).

    One pin — ``REQUEST_TIMEOUT`` — is not cognee's own setting but LiteLLM's,
    and unlike the rest it reaches every LiteLLM caller in the process rather
    than just this session's engine. That is deliberate and, in the
    process-scoped bake-off, harmless; see the module docstring.

    Args:
        workdir: The session's working directory (the embedded stores' home).

    Returns:
        Every environment variable :func:`_pin_env` writes.
    """
    settings = get_settings()
    model = settings.BASE_MODEL
    if "/" not in model:
        # The custom provider routes chat calls through LiteLLM, which
        # requires a provider-prefixed model name and rejects a bare one
        # outright ("LLM Provider NOT provided" — verified live, Phase 3
        # gate). "openai/" selects its OpenAI-compatible handler against
        # LLM_ENDPOINT; llama.cpp itself ignores the name (single-model
        # server). A value that already carries a prefix passes untouched.
        model = f"openai/{model}"
    return {
        "LLM_PROVIDER": "custom",
        "LLM_MODEL": model,
        "LLM_ENDPOINT": settings.BASE_MODEL_API_URL,
        "LLM_API_KEY": settings.BASE_MODEL_API_KEY,
        # LiteLLM's completion deadline. Its 600 s fallback is a *client*
        # timeout, so on a single-slot server it fires on the queue rather
        # than on a hang: at the shipped fan-out the tail extraction call
        # waited past it, instructor re-sent it, and the retry storm killed a
        # 76-minute build with zero entities (2026-07-28, first full-profile
        # run). Bounded fan-out plus this deadline make that unreachable.
        "REQUEST_TIMEOUT": str(_REQUEST_TIMEOUT_S),
        # cognee pre-flights one structured LLM call with a hard 30 s cap
        # before every first pipeline run. On a single-slot llama.cpp a slow
        # first token would fail an entire build over a guardrail, and a
        # genuinely dead endpoint surfaces immediately in the add/cognify
        # failure counts anyway (criterion §8.2#2 records them).
        "COGNEE_SKIP_CONNECTION_TEST": "true",
        # Single-user bake-off: cognee 1.x defaults to multi-tenant access
        # control, which wraps search results per dataset and scopes the
        # graph away from the default stats context (observed live: a graph
        # that answered correctly read back as 0 nodes — Phase 3 gate). The
        # repo is single-user by design, so cognee runs the same way.
        "ENABLE_BACKEND_ACCESS_CONTROL": "false",
        # Session memory (on by default since 1.0) persists in the workdir
        # and folds earlier answers into later ones ("I've already answered
        # that…" — observed live, across processes). Cross-question memory
        # would contaminate golden scoring, so the bake-off runs without it.
        "CACHING": "false",
        "EMBEDDING_PROVIDER": "openai_compatible",
        "EMBEDDING_MODEL": settings.EMBEDDING_MODEL,
        # The direct-SDK path appends /v1 itself, so the endpoint is given
        # without it (settings.EMBEDDING_API_URL carries the suffix).
        "EMBEDDING_ENDPOINT": settings.EMBEDDING_API_URL.removesuffix("/v1").rstrip("/"),
        "EMBEDDING_API_KEY": settings.EMBEDDING_API_KEY,
        "EMBEDDING_DIMENSIONS": str(_EMBEDDING_DIM),
        "DATA_ROOT_DIRECTORY": str(workdir / "data"),
        "SYSTEM_ROOT_DIRECTORY": str(workdir / "system"),
    }


def _triplets(retrieved: Any) -> Iterator[_Triplet]:
    """Yield the ``(source, edge, target)`` mappings from a triplet-shaped payload.

    Args:
        retrieved: The evidence-pass search results.

    Yields:
        Each well-formed triplet; anything else is skipped.
    """
    if not isinstance(retrieved, Sequence) or isinstance(retrieved, str):
        return
    for item in _result_items(retrieved):
        if not _is_triplet(item):
            continue
        source, edge, target = (_jsonable(part) for part in item)
        if (
            isinstance(source, Mapping)
            and isinstance(edge, Mapping)
            and isinstance(target, Mapping)
        ):
            yield source, edge, target


def _is_triplet(item: Any) -> bool:
    """Report whether a results item is a three-element sequence.

    Args:
        item: One results entry.

    Returns:
        ``True`` for a ``(source, edge, target)``-shaped entry.
    """
    return isinstance(item, Sequence) and not isinstance(item, str) and len(item) == 3


def _jsonable(value: Any) -> Any:
    """Coerce a cognee model (or anything else) into plain JSON-able data.

    Args:
        value: A pydantic model, mapping, sequence, or scalar.

    Returns:
        The plain-data equivalent; scalars and unknown objects come back
        stringified so a payload never blocks serialization.
    """
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        dumped = dump()
        return dumped if isinstance(dumped, dict | list) else str(dumped)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_jsonable(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _text(item: Mapping[str, Any], *names: str) -> str | None:
    """Read the first present non-empty string field from a payload item.

    Args:
        item: One node or edge record.
        *names: Candidate field names, most likely first.

    Returns:
        The trimmed value, or ``None`` when no candidate holds text.
    """
    for name in names:
        value = item.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None
