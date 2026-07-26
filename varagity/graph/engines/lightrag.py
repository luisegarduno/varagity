"""The LightRAG bake-off adapter (spec_graphrag §8; ADR-017 candidate).

LightRAG is the document-shaped candidate with the cleanest injection points:
a custom async ``llm_model_func`` and an ``EmbeddingFunc(embedding_dim=1024)``
are first-class constructor arguments, so this is the one engine where the
``<think>``-strip template applies **fully** — every extraction, gleaning, and
answer completion passes through :func:`~varagity.models.llm.clean_response`
before LightRAG parses it. A reasoning model's unstripped ``<think>`` block
breaks its delimiter grammar silently, which is exactly the trap the condense
and HyDE features already paid for.

Storage is the file-based default (``NetworkXStorage`` + ``NanoVectorDBStorage``
+ JSON KV/doc-status) inside the session's working directory: LightRAG's
Postgres graph plane needs the Apache AGE extension (R1 correction), which is a
different image and therefore an ADR-017 storage question, not a stage-1 one.

Two deliberate deviations, both recorded rather than hidden (spec §6):

* **e5's asymmetric prefix is not applied.** LightRAG calls one embedding
  function for passages and queries alike, so the query-side
  ``Instruct: … / Query: …`` wrapper
  (:func:`varagity.models.embeddings.format_query`) has no seam to hook. The
  bake-off runs unprefixed for every document-shaped engine and the deviation
  goes in the results, rather than being hacked around per engine.
* **Concurrency knobs are pinned through ``os.environ``.** They are read at
  import time by LightRAG's own settings, so they must be set *before* the
  lazy import — the one place in this repo where writing environment
  variables is the mechanism rather than a smell (varagity's own config is
  still read only through :func:`~varagity.config.get_settings`).
"""

import asyncio
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

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
    doc_guid_index,
    guids_in_payload,
    merge_batches,
    thread_transcripts,
)
from varagity.graph.sources.base import MessageBatch
from varagity.models.llm import clean_response
from varagity.tokens import count_tokens

logger = logging.getLogger(__name__)

# LightRAG's own query modes: local / global / hybrid / naive / mix. `hybrid`
# fuses its dual-level (entity + relation) keyword retrieval and is the
# adapter's primary; `--mode global` is the recorded extra pass the
# aggregation questions get (plan decision #13).
PRIMARY_MODE = "hybrid"

# e5's dimensionality — load-bearing for LightRAG, whose EmbeddingFunc
# validates that every returned vector divides by it.
_EMBEDDING_DIM = 1024

# Generation cap per engine call, then clamped against the context window.
# Extraction replies are structured records, not prose, so a modest cap keeps
# a reasoning model from spending the whole window in <think>.
_MAX_TOKENS = 2048
_MIN_TOKENS = 256
# Headroom for the chat template's scaffolding plus the cl100k approximation's
# drift versus the served model's tokenizer (the varagity.models.llm constant).
_CTX_HEADROOM_TOKENS = 512

# Generous per-call HTTP timeouts: a single-slot llama.cpp queues extraction
# calls, so the wait is the queue, not a hang (LightRAG's own LLM_TIMEOUT
# otherwise falls back to its gunicorn worker's 150 s — a documented trap).
_LLM_TIMEOUT_S = 1800.0
_EMBEDDING_TIMEOUT_S = 600.0

# Documents per insert call. LightRAG resumes from its doc-status store, so
# chunking the insert turns a mid-run failure into a recorded partial rather
# than a lost multi-hour index.
_INSERT_BATCH = 20

# Read by LightRAG at import time, so they are pinned before the lazy import.
# Storage classes are pinned to the file-based defaults so a stray environment
# on the machine cannot silently redirect the bake-off at a real database.
_ENV_PINS: dict[str, str] = {
    "MAX_ASYNC_LLM": "1",
    "MAX_PARALLEL_INSERT": "1",
    "EMBEDDING_FUNC_MAX_ASYNC": "2",
    "LLM_TIMEOUT": str(int(_LLM_TIMEOUT_S)),
    "EMBEDDING_TIMEOUT": str(int(_EMBEDDING_TIMEOUT_S)),
    "LIGHTRAG_KV_STORAGE": "JsonKVStorage",
    "LIGHTRAG_VECTOR_STORAGE": "NanoVectorDBStorage",
    "LIGHTRAG_GRAPH_STORAGE": "NetworkXStorage",
    "LIGHTRAG_DOC_STATUS_STORAGE": "JsonDocStatusStorage",
}


def fit_max_tokens(messages: Sequence[Mapping[str, Any]], cap: int, context_tokens: int) -> int:
    """Clamp a generation cap so prompt + generation fits the context window.

    The same discipline :func:`varagity.models.llm._fit_max_tokens` applies to
    the app's own client, re-implemented here because LightRAG is handed a raw
    async callable rather than an :class:`~varagity.models.llm.LLMClient`:
    llama.cpp with context shift disabled hard-500s mid-decode at the window
    instead of stopping gracefully, and LightRAG's extraction prompts are
    large.

    Args:
        messages: The chat messages about to be sent (string contents are
            counted).
        cap: The requested generation cap.
        context_tokens: The served model's context window
            (``LLM_CONTEXT_TOKENS``).

    Returns:
        The cap, reduced to fit. A prompt that leaves no room at all still
        gets :data:`_MIN_TOKENS`: that request is already lost, and a floor
        keeps the failure legible in the engine's own error rather than in a
        zero-token request. A caller asking for less than the floor still
        gets what it asked for.
    """
    used = sum(
        count_tokens(content)
        for message in messages
        if isinstance(content := message.get("content"), str)
    )
    available = context_tokens - used - _CTX_HEADROOM_TOKENS
    if available < cap:
        logger.debug("clamping LightRAG generation cap %d → %d (prompt ~%d)", cap, available, used)
    return min(cap, max(_MIN_TOKENS, available))


def make_llm_func(
    client: AsyncOpenAI,
    *,
    model: str,
    temperature: float,
    context_tokens: int,
) -> Callable[..., Awaitable[str]]:
    """Build the async completion callable LightRAG drives every LLM stage with.

    Args:
        client: An async OpenAI-compatible client pointed at llama.cpp.
        model: Model name to send (llama.cpp echoes it — provenance, not
            routing).
        temperature: Sampling temperature.
        context_tokens: The served model's context window, for the clamp.

    Returns:
        A coroutine function with LightRAG's expected signature
        (``prompt``, optional ``system_prompt``/``history_messages``, plus
        whatever keyword arguments the calling stage passes) returning
        ``<think>``-stripped text.
    """

    async def llm_model_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: Sequence[Mapping[str, Any]] | None = None,
        **_kwargs: Any,
    ) -> str:
        messages: list[ChatCompletionMessageParam] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        # LightRAG's history entries are already OpenAI-shaped dicts; the cast
        # is the same duck-typing concession the chat engines make at the LLM
        # seam (varagity/chat/condense.py).
        messages.extend(
            cast("ChatCompletionMessageParam", dict(turn)) for turn in history_messages or ()
        )
        messages.append({"role": "user", "content": prompt})
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=fit_max_tokens(messages, _MAX_TOKENS, context_tokens),
            temperature=temperature,
        )
        # Mandatory: LightRAG parses delimiter-grammar records straight out of
        # this string, and an unstripped reasoning stage breaks that silently
        # (the condense/HyDE precedent, spec_graphrag §7).
        return clean_response(response.choices[0].message.content or "")

    return llm_model_func


def make_embedding_func(
    client: AsyncOpenAI, *, model: str
) -> Callable[[Sequence[str]], Awaitable[list[list[float]]]]:
    """Build the async embedding callable LightRAG embeds documents and queries with.

    Passages are sent unprefixed, which is correct for e5 passages and the
    recorded deviation for e5 queries (see the module docstring).

    Args:
        client: An async OpenAI-compatible client pointed at infinity.
        model: Served embedding model name.

    Returns:
        A coroutine function mapping texts to their 1024-dim vectors, in order.
    """

    async def embed(texts: Sequence[str]) -> list[list[float]]:
        response = await client.embeddings.create(model=model, input=list(texts))
        return [item.embedding for item in response.data]

    return embed


def evidence_from_context(context: Any, index: Mapping[str, Sequence[str]]) -> GraphEvidence:
    """Normalize LightRAG's retrieval context into engine-independent evidence.

    LightRAG returns its context as JSON (or as a formatted string, depending
    on version and mode), so the mapping is alias-tolerant and every unknown
    shape degrades to ``raw`` plus empty lists rather than raising — a
    normalizer that throws would turn an answered question into a harness
    crash.

    Args:
        context: Whatever ``aquery(..., only_need_context=True)`` returned.
        index: ``doc_key`` → message guids, for provenance recovery.

    Returns:
        The normalized evidence. LightRAG has no community layer, so
        ``communities`` is always empty.
    """
    raw = _as_mapping(context)
    walked = context if raw is None else raw
    return GraphEvidence(
        entities=[
            entity for item in _section(raw, "entities") if (entity := _entity(item)) is not None
        ],
        relations=[
            relation
            for item in _section(raw, "relationships", "relations")
            if (relation := _relation(item)) is not None
        ],
        communities=[],
        message_guids=guids_in_payload(walked, index),
        raw=raw,
    )


@register("lightrag")
class LightRAGEngine:
    """LightRAG behind the :class:`~varagity.graph.base.GraphEngine` protocol."""

    @contextmanager
    def session(self, workdir: Path) -> Iterator[GraphSession]:
        """Open a LightRAG session storing everything under ``workdir``.

        Args:
            workdir: The engine's working directory (created if absent).

        Yields:
            The open session; storages are initialized on entry and finalized
            on exit.
        """
        settings = get_settings()
        _pin_env()
        # Lazy, so the registry costs nothing without the bakeoff group
        # installed (plan decision #8) — numpy included: LightRAG's
        # EmbeddingFunc expects an array, and numpy is not a declared
        # dependency of this project, only a transitive one.
        import numpy as np
        from lightrag import LightRAG, QueryParam
        from lightrag.kg.shared_storage import initialize_pipeline_status
        from lightrag.utils import EmbeddingFunc

        workdir.mkdir(parents=True, exist_ok=True)
        chat_client = AsyncOpenAI(
            base_url=settings.BASE_MODEL_API_URL,
            api_key=settings.BASE_MODEL_API_KEY,
            timeout=_LLM_TIMEOUT_S,
        )
        embedding_client = AsyncOpenAI(
            base_url=settings.EMBEDDING_API_URL,
            api_key=settings.EMBEDDING_API_KEY,
            timeout=_EMBEDDING_TIMEOUT_S,
        )
        embed = make_embedding_func(embedding_client, model=settings.EMBEDDING_MODEL)

        async def embedding_func(texts: Sequence[str]) -> Any:
            return np.array(await embed(texts), dtype=np.float32)

        rag = LightRAG(
            working_dir=str(workdir),
            llm_model_func=make_llm_func(
                chat_client,
                model=settings.BASE_MODEL,
                temperature=settings.LLM_TEMPERATURE,
                context_tokens=settings.LLM_CONTEXT_TOKENS,
            ),
            embedding_func=EmbeddingFunc(embedding_dim=_EMBEDDING_DIM, func=embedding_func),
        )
        session = _LightRAGSession(rag, QueryParam)
        try:
            session.run(rag.initialize_storages())
            session.run(initialize_pipeline_status())
            yield session
        finally:
            session.close()


class _LightRAGSession:
    """One open LightRAG working session (see :class:`LightRAGEngine`)."""

    def __init__(
        self, rag: Any, query_param: Callable[..., Any], *, mode: str = PRIMARY_MODE
    ) -> None:
        """Wrap an initialized LightRAG instance.

        Args:
            rag: The ``LightRAG`` instance (injected so the session's logic is
                testable against a double, with only :meth:`LightRAGEngine.session`
                touching the real library).
            query_param: LightRAG's ``QueryParam`` class, called with
                ``mode=`` / ``only_need_context=``.
            mode: Primary query mode for this session.
        """
        self._rag = rag
        self._query_param = query_param
        self._mode = mode
        self._index: dict[str, list[str]] = {}
        # One event loop per session: LightRAG's async primitives bind to the
        # running loop, so a fresh asyncio.run() per call would strand them.
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
        """Index the batches' merged messages as thread transcripts.

        Documents are inserted under their :attr:`~varagity.graph.render.TranscriptDoc.doc_key`,
        which is LightRAG's document id: re-inserting an unchanged transcript
        is a doc-status hit, so an overlapping upload or a second build costs
        nothing and duplicates nothing.

        Args:
            batches: Parsed source files (guid-merged before rendering).
            verbose: Validated console verbosity (0–2).

        Returns:
            The build report; a failed insert chunk is recorded and the rest
            of the corpus still indexes.
        """
        messages = merge_batches(batches)
        docs = thread_transcripts(messages)
        self._index.update(doc_guid_index(docs))
        failures: list[str] = []
        started = time.perf_counter()
        for start in range(0, len(docs), _INSERT_BATCH):
            chunk = docs[start : start + _INSERT_BATCH]
            try:
                self.run(
                    self._rag.ainsert(
                        [doc.text for doc in chunk],
                        ids=[doc.doc_key for doc in chunk],
                        file_paths=[doc.doc_key for doc in chunk],
                    )
                )
            except Exception as exc:  # a failed chunk is data, not the end of the run
                logger.warning("LightRAG insert failed for %d document(s)", len(chunk))
                failures.append(f"insert {chunk[0].doc_key}…(+{len(chunk) - 1}): {exc!r}")
        return BuildReport(
            messages_seen=len(messages),
            wall_clock_s=time.perf_counter() - started,
            failures=failures,
        )

    def query(self, question: str, *, mode: str | None = None, verbose: int = 0) -> GraphAnswer:
        """Answer one question with LightRAG's own answer pipeline.

        Retrieval runs twice: once for the answer and once with
        ``only_need_context=True`` for the evidence. The second pass costs
        retrieval only (no generation), which is the cheapest honest way to
        report what the answer was built from.

        Args:
            question: The question, verbatim.
            mode: LightRAG query mode; ``None`` uses the session's primary.
            verbose: Validated console verbosity (0–2).

        Returns:
            The answer with its normalized evidence.
        """
        used = mode or self._mode
        started = time.perf_counter()
        answer = self.run(self._rag.aquery(question, param=self._query_param(mode=used)))
        context = self._context(question, used)
        return GraphAnswer(
            answer=clean_response(str(answer)),
            evidence=evidence_from_context(context, self._index),
            mode=used,
            latency_s=time.perf_counter() - started,
        )

    def stats(self) -> GraphStats:
        """Count the graph's nodes and edges from LightRAG's graph storage.

        Returns:
            Entity/relation counts, or ``None`` for each if the storage would
            not say. ``communities`` is always ``None``: LightRAG builds no
            community layer at all (R1), which is itself an ADR-017 datum.
        """
        entities = relations = None
        try:
            graph = self.run(self._rag.chunk_entity_relation_graph.get_knowledge_graph("*"))
            entities = len(graph.nodes)
            relations = len(graph.edges)
        except Exception:  # an engine that won't report is reported as unknown
            logger.warning("LightRAG graph stats unavailable", exc_info=True)
        return GraphStats(entities=entities, relations=relations, communities=None)

    def close(self) -> None:
        """Finalize LightRAG's storages and close the session's event loop."""
        try:
            self.run(self._rag.finalize_storages())
        except Exception:
            logger.warning("LightRAG teardown failed", exc_info=True)
        finally:
            self._loop.close()

    def _context(self, question: str, mode: str) -> Any:
        """Retrieve the context behind an answer, tolerating retrieval failure.

        Args:
            question: The question, verbatim.
            mode: The query mode used for the answer.

        Returns:
            LightRAG's context payload, or ``None`` if it could not be
            retrieved (the answer still stands; the evidence is empty).
        """
        try:
            return self.run(
                self._rag.aquery(
                    question, param=self._query_param(mode=mode, only_need_context=True)
                )
            )
        except Exception:
            logger.warning("LightRAG context retrieval failed", exc_info=True)
            return None


def _pin_env() -> None:
    """Pin LightRAG's import-time environment knobs to single-slot values."""
    os.environ.update(_ENV_PINS)


def _as_mapping(context: Any) -> dict[str, Any] | None:
    """Coerce LightRAG's context payload into a mapping, if it is one.

    Args:
        context: The raw context (mapping, JSON string, prose, or ``None``).

    Returns:
        The mapping, a single-key ``{"context": …}`` wrapper for a non-JSON
        string, or ``None`` when there is nothing to keep.
    """
    if isinstance(context, Mapping):
        return dict(context)
    if isinstance(context, str) and context.strip():
        try:
            parsed = json.loads(context)
        except ValueError:
            return {"context": context}
        return dict(parsed) if isinstance(parsed, Mapping) else {"context": context}
    return None


def _section(raw: Mapping[str, Any] | None, *names: str) -> list[Mapping[str, Any]]:
    """Read the first present list-of-mappings section from a context payload.

    Args:
        raw: The context mapping, or ``None``.
        *names: Candidate key names, most likely first.

    Returns:
        The section's mapping items (non-mapping entries are skipped), or an
        empty list when no candidate key holds a list.
    """
    if raw is None:
        return []
    for name in names:
        value = raw.get(name)
        if isinstance(value, Sequence) and not isinstance(value, str):
            return [item for item in value if isinstance(item, Mapping)]
    return []


def _text(item: Mapping[str, Any], *names: str) -> str | None:
    """Read the first present non-empty string field from a payload item.

    Args:
        item: One entity/relation record.
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
    """Map one LightRAG entity record.

    Args:
        item: The record.

    Returns:
        The entity, or ``None`` when it carries no usable name.
    """
    name = _text(item, "entity_name", "entity", "name", "id")
    if name is None:
        return None
    return GraphEntity(
        name=name,
        type=_text(item, "entity_type", "type", "category"),
        summary=_text(item, "description", "summary", "content"),
    )


def _relation(item: Mapping[str, Any]) -> GraphRelation | None:
    """Map one LightRAG relationship record.

    Args:
        item: The record.

    Returns:
        The relation, or ``None`` when either endpoint is missing.
    """
    source = _text(item, "src_id", "source", "src", "entity1")
    target = _text(item, "tgt_id", "target", "tgt", "entity2")
    if source is None or target is None:
        return None
    return GraphRelation(
        source=source,
        target=target,
        label=_text(item, "keywords", "label", "relation", "type"),
        description=_text(item, "description", "summary", "content"),
    )
