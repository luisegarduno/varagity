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

Two more things this adapter owns, both recorded rather than hidden (spec §6):

* **e5's asymmetric prefix is opt-in behind ``GRAPH_QUERY_PREFIX``.** The
  bake-off ran unprefixed (a recorded stage-1 deviation) because LightRAG
  appeared to call one embedding function for passages and queries alike. It
  does not: ``EmbeddingFunc(supports_asymmetric=True)`` forwards a
  ``context="document"|"query"`` keyword to the hook
  (``lightrag/utils.py:596-602``), and the pinned vector store passes it on
  every search (``kg/nano_vector_db_impl.py:429-431``; also
  ``operate.py:4382`` for the dual-level keyword embeddings). Passages stay
  unprefixed either way — which is *correct* under e5 discipline — so the
  setting is measurable on an already-built graph without re-embedding
  anything. :func:`varagity.models.embeddings.format_query` stays the single
  owner of the query format.
* **Concurrency knobs are pinned through ``os.environ``.** They are read at
  import time by LightRAG's own settings, so they must be set *before* the
  lazy import — the one place in this repo where writing environment
  variables is the mechanism rather than a smell (varagity's own config is
  still read only through :func:`~varagity.config.get_settings`).

Answer composition is **ours, not the engine's**, on any ``<base>+synthesis``
mode (ADR-017's retrieval-only decision): one ``aquery_data`` call returns
structured entities/relations/chunks with no LLM answer stage, and
:mod:`varagity.graph.answer` writes the grounded answer over them. Unsuffixed
modes keep the bake-off's engine-composed path verbatim, so ADR-017's numbers
stay reproducible.
"""

import asyncio
import json
import logging
import os
import re
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from varagity.config import get_settings
from varagity.graph.answer import synthesis_context, synthesis_max_chars, synthesize
from varagity.graph.base import GraphSession, register
from varagity.graph.records import (
    BuildReport,
    GraphAnswer,
    GraphEntity,
    GraphEvidence,
    GraphRelation,
    GraphStats,
    TranscriptExcerpt,
)
from varagity.graph.render import (
    doc_guid_index,
    guids_in_payload,
    merge_batches,
    thread_transcripts,
)
from varagity.graph.sources.base import MessageBatch
from varagity.models.embeddings import format_query
from varagity.models.llm import LLMClient, clean_response
from varagity.tokens import count_tokens

logger = logging.getLogger(__name__)

# LightRAG's own query modes: local / global / hybrid / naive / mix. `hybrid`
# fuses its dual-level (entity + relation) keyword retrieval and is the
# adapter's primary; `--mode global` is the recorded extra pass the
# aggregation questions get (plan decision #13).
PRIMARY_MODE = "hybrid"

# Mode suffix selecting ADR-017's retrieval-only design: `hybrid+synthesis`
# retrieves with `hybrid` and lets varagity.graph.answer write the answer.
# A bare `synthesis` means "the session's primary base mode, synthesized".
_SYNTHESIS_SUFFIX = "+synthesis"
_SYNTHESIS_BARE = _SYNTHESIS_SUFFIX.removeprefix("+")

# The value LightRAG's query paths pass to an asymmetric embedding hook
# (verified in 1.5.4's sources — see the module docstring).
_QUERY_CONTEXT = "query"

# The transcript header render.py writes at the top of every document; a
# retrieved chunk that still carries it names its thread far better than the
# thread *id* parsed out of the doc_key can.
_THREAD_HEADER_RE = re.compile(r"^Thread:\s*(.+?)\s*\(participants:", re.MULTILINE)

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
    client: AsyncOpenAI, *, model: str, query_prefix: bool = False
) -> Callable[..., Awaitable[list[list[float]]]]:
    """Build the async embedding callable LightRAG embeds documents and queries with.

    Passages are always sent unprefixed, which is what e5 discipline requires
    of documents. Queries are wrapped with
    :func:`~varagity.models.embeddings.format_query` only when
    ``query_prefix`` is on *and* the caller declared the call's side — which
    LightRAG does through the ``context`` keyword its
    ``EmbeddingFunc(supports_asymmetric=True)`` wrapper forwards. Both gates
    matter: with ``supports_asymmetric`` off the wrapper strips ``context``
    before this function ever sees it, so the flag has to reach both.

    Args:
        client: An async OpenAI-compatible client pointed at infinity.
        model: Served embedding model name.
        query_prefix: Whether to apply e5's query-side instruction wrapper
            (``GRAPH_QUERY_PREFIX``).

    Returns:
        A coroutine function mapping texts to their 1024-dim vectors, in
        order, accepting LightRAG's optional ``context`` keyword.
    """

    async def embed(texts: Sequence[str], context: str | None = None) -> list[list[float]]:
        payload = list(texts)
        if query_prefix and context == _QUERY_CONTEXT:
            payload = [format_query(text) for text in payload]
        response = await client.embeddings.create(model=model, input=payload)
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


def retrieval_from_query_data(
    payload: Any, index: Mapping[str, Sequence[str]]
) -> tuple[GraphEvidence, list[TranscriptExcerpt]]:
    """Normalize an ``aquery_data`` payload into evidence plus transcript excerpts.

    ``aquery_data`` is the structured retrieval API: it forces
    ``only_need_context`` internally and returns
    ``{"status", "data": {"entities", "relationships", "chunks",
    "references"}, "metadata"}`` with **no answer call**, which is exactly
    what ADR-017's retrieval-only design needs. A failure payload carries an
    empty ``data``, and — like every normalizer here — an unknown shape
    degrades to empty lists rather than raising.

    Args:
        payload: Whatever ``aquery_data`` returned (or ``None``).
        index: ``doc_key`` → message guids, for provenance recovery.

    Returns:
        The evidence (entities/relations; LightRAG has no community layer)
        and the retrieved transcript passages, in the engine's own relevance
        order.
    """
    raw = _as_mapping(payload)
    data = _data_section(raw)
    evidence = GraphEvidence(
        entities=[
            entity for item in _section(data, "entities") if (entity := _entity(item)) is not None
        ],
        relations=[
            relation
            for item in _section(data, "relationships", "relations")
            if (relation := _relation(item)) is not None
        ],
        communities=[],
        message_guids=guids_in_payload(payload if raw is None else raw, index),
        raw=raw,
    )
    excerpts = [
        excerpt
        for item in _section(data, "chunks")
        if (excerpt := _excerpt(item, index)) is not None
    ]
    return evidence, excerpts


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
        asymmetric = settings.GRAPH_QUERY_PREFIX
        if asymmetric:
            logger.info(
                "GRAPH_QUERY_PREFIX on: queries are e5-instruction-wrapped "
                "(supports_asymmetric=True); passages stay unprefixed"
            )
        embed = make_embedding_func(
            embedding_client, model=settings.EMBEDDING_MODEL, query_prefix=asymmetric
        )

        async def embedding_func(texts: Sequence[str], context: str | None = None) -> Any:
            return np.array(await embed(texts, context=context), dtype=np.float32)

        rag = LightRAG(
            working_dir=str(workdir),
            llm_model_func=make_llm_func(
                chat_client,
                model=settings.BASE_MODEL,
                temperature=settings.LLM_TEMPERATURE,
                context_tokens=settings.LLM_CONTEXT_TOKENS,
            ),
            embedding_func=EmbeddingFunc(
                embedding_dim=_EMBEDDING_DIM,
                func=embedding_func,
                # Without this LightRAG strips `context` before calling the
                # hook above, and the prefix setting is a dead knob.
                supports_asymmetric=asymmetric,
            ),
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
        self,
        rag: Any,
        query_param: Callable[..., Any],
        *,
        mode: str = PRIMARY_MODE,
        llm: LLMClient | None = None,
    ) -> None:
        """Wrap an initialized LightRAG instance.

        Args:
            rag: The ``LightRAG`` instance (injected so the session's logic is
                testable against a double, with only :meth:`LightRAGEngine.session`
                touching the real library).
            query_param: LightRAG's ``QueryParam`` class, called with
                ``mode=`` / ``only_need_context=``.
            mode: Primary query mode for this session.
            llm: Chat client for ``+synthesis`` answers; ``None`` builds one
                on first use, so a session that only ever runs engine-composed
                modes never opens the connection.
        """
        self._rag = rag
        self._query_param = query_param
        self._mode = mode
        self._llm = llm
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
        """Answer one question, with either the engine's pipeline or ours.

        A ``<base>+synthesis`` mode runs ADR-017's shipped design: one
        ``aquery_data`` retrieval (structured, no engine answer call) and a
        grounded answer written by :mod:`varagity.graph.answer` over the
        entities, relations, **and** transcript excerpts it returned.

        An unsuffixed mode keeps the bake-off's engine-composed path
        verbatim: retrieval runs twice, once for LightRAG's own answer and
        once with ``only_need_context=True`` for the evidence (the second
        pass costs retrieval only). ADR-017's numbers stay reproducible.

        Args:
            question: The question, verbatim.
            mode: LightRAG query mode, optionally ``+synthesis``-suffixed;
                ``None`` uses the session's primary.
            verbose: Validated console verbosity (0–2).

        Returns:
            The answer with its normalized evidence, recorded under the full
            mode string that produced it.
        """
        used = mode or self._mode
        base, wants_synthesis = _split_mode(used)
        if not base:
            base = _split_mode(self._mode)[0] or PRIMARY_MODE
        started = time.perf_counter()
        if wants_synthesis:
            evidence, excerpts = retrieval_from_query_data(
                self._query_data(question, base), self._index
            )
            context = synthesis_context(
                evidence, excerpts, max_chars=synthesis_max_chars(get_settings().LLM_CONTEXT_TOKENS)
            )
            answer_text = synthesize(self._llm_client(), question, context)
            return GraphAnswer(
                answer=answer_text,
                evidence=evidence,
                mode=used,
                latency_s=time.perf_counter() - started,
            )
        answer = self.run(self._rag.aquery(question, param=self._query_param(mode=base)))
        context_payload = self._context(question, base)
        return GraphAnswer(
            answer=clean_response(str(answer)),
            evidence=evidence_from_context(context_payload, self._index),
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

    def _llm_client(self) -> LLMClient:
        """Return the session's chat client, building it on first use.

        Returns:
            The client used for ``+synthesis`` answers.
        """
        if self._llm is None:
            self._llm = LLMClient()
        return self._llm

    def _query_data(self, question: str, mode: str) -> Any:
        """Retrieve structured evidence, tolerating retrieval failure.

        Args:
            question: The question, verbatim.
            mode: The base query mode (no ``+synthesis`` suffix).

        Returns:
            LightRAG's ``aquery_data`` payload, or ``None`` if it could not be
            retrieved — which synthesizes to the honest "no facts" answer
            rather than taking the run down.
        """
        try:
            return self.run(self._rag.aquery_data(question, param=self._query_param(mode=mode)))
        except Exception:
            logger.warning("LightRAG structured retrieval failed", exc_info=True)
            return None

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


def _split_mode(mode: str) -> tuple[str, bool]:
    """Split a query mode into its base engine mode and the synthesis flag.

    Args:
        mode: A mode name, optionally ``+synthesis``-suffixed
            (``"hybrid+synthesis"``), or the bare ``"synthesis"``.

    Returns:
        ``(base_mode, wants_synthesis)``. The base is ``""`` when the caller
        named no mode of its own (``"synthesis"`` / ``"+synthesis"``), which
        the session reads as "my primary mode".
    """
    stem = mode.strip()
    if stem == _SYNTHESIS_BARE:
        return "", True
    if stem.endswith(_SYNTHESIS_SUFFIX):
        return stem[: -len(_SYNTHESIS_SUFFIX)].strip(), True
    return stem, False


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


def _data_section(raw: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """Descend into an ``aquery_data`` payload's ``data`` envelope.

    Args:
        raw: The payload mapping, or ``None``.

    Returns:
        The ``data`` sub-mapping when the payload is enveloped, the payload
        itself when the sections sit at the top level (the ``aquery`` context
        shape), or ``None``.
    """
    if raw is None:
        return None
    data = raw.get("data")
    return data if isinstance(data, Mapping) else raw


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


def _excerpt(
    item: Mapping[str, Any], index: Mapping[str, Sequence[str]]
) -> TranscriptExcerpt | None:
    """Map one retrieved chunk into a transcript excerpt.

    The adapter inserts every document under its ``doc_key`` as both id and
    ``file_path``, so a chunk's ``file_path`` *is* the corpus join key — and
    provenance resolves through the same tolerant walk the rest of the
    adapter uses (LightRAG joins multiple paths with its own separator).

    Args:
        item: One ``chunks`` record.
        index: ``doc_key`` → message guids.

    Returns:
        The excerpt, or ``None`` when the chunk carries no text (a passage
        with nothing in it cannot ground anything).
    """
    text = _text(item, "content", "text", "chunk")
    if text is None:
        return None
    doc_key = _text(item, "file_path", "doc_key", "file_paths", "source") or ""
    thread_id, _, span = doc_key.partition("::")
    header = _THREAD_HEADER_RE.search(text)
    return TranscriptExcerpt(
        doc_key=doc_key,
        thread_name=header.group(1) if header else thread_id,
        span=span,
        text=text,
        message_guids=guids_in_payload(doc_key, index),
    )
