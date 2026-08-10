"""The LightRAG adapter — the shipped graph engine (spec_graphrag §8; ADR-017).

LightRAG is the document-shaped engine with the cleanest injection points:
a custom async ``llm_model_func`` and an ``EmbeddingFunc(embedding_dim=1024)``
are first-class constructor arguments, so this is an engine where the
``<think>``-strip template applies **fully** — every extraction, gleaning, and
answer completion passes through :func:`~varagity.models.llm.clean_response`
before LightRAG parses it. A reasoning model's unstripped ``<think>`` block
breaks its delimiter grammar silently, which is exactly the trap the condense
and HyDE features already paid for.

Storage is the file-based default (``NetworkXStorage`` + ``NanoVectorDBStorage``
+ JSON KV/doc-status) inside the session's working directory (ADR-017 pinned
all four classes): LightRAG's Postgres graph plane needs the Apache AGE
extension, which is a different image than this stack runs.

**One process, one writer, queries live throughout.** The session drives a
private event loop on a daemon thread and submits every engine call with
``asyncio.run_coroutine_threadsafe``, so a chat query is answered while a
multi-day backfill is still extracting — which is how LightRAG's own server
behaves, and why :class:`~varagity.graph.service.GraphService` keeps exactly
one session per process (stage-2 decisions #10/#11). The engine's storages
are single-writer per workspace by explicit invariant, so nothing outside
this process may write the workdir; that is the reason there is no CLI graph
build.

**Builds are diffing upserts, not appends.** LightRAG's enqueue stage drops
a re-submitted ``doc_id`` in *any* status, so a re-exported ``chat.db`` whose
newest thread-day grew would silently keep the stale transcript.
:mod:`varagity.graph.manifest` records what was indexed; :meth:`_LightRAGSession.build`
deletes the genuinely changed documents before re-enqueueing them and pays
extraction for nothing else. Enqueue and process are separate engine calls,
and process re-selects every in-flight/failed document at the top of each
batch — so a killed build resumes by calling :meth:`_LightRAGSession.resume`.

Two more things this adapter owns, both recorded rather than hidden (spec §6):

* **e5's asymmetric prefix rides ``GRAPH_QUERY_PREFIX``.** The bake-off ran
  unprefixed (a recorded stage-1 deviation) because LightRAG appeared to call
  one embedding function for passages and queries alike. It does not:
  ``EmbeddingFunc(supports_asymmetric=True)`` forwards a
  ``context="document"|"query"`` keyword to the hook
  (``lightrag/utils.py:596-602``), and the pinned vector store passes it on
  every search (``kg/nano_vector_db_impl.py:429-431``; also
  ``operate.py:4382`` for the dual-level keyword embeddings). Passages stay
  unprefixed either way — which is *correct* under e5 discipline — so the
  setting was measurable on the already-built bake-off graph without
  re-embedding anything, and the acceptance gate shipped it **on**.
  :func:`varagity.models.embeddings.format_query` stays the single owner of
  the query format.
* **Concurrency knobs are pinned through ``os.environ``.** They are read at
  import time by LightRAG's own settings, so they must be set *before* the
  lazy import — the one place in this repo where writing environment
  variables is the mechanism rather than a smell (varagity's own config is
  still read only through :func:`~varagity.config.get_settings`).

Answer composition is **ours, not the engine's** (ADR-017's retrieval-only
decision). :meth:`_LightRAGSession.retrieve` is that half on its own: one
``aquery_data`` call returning structured entities/relations/chunks with no
LLM answer stage. The app streams its answer over it
(:func:`varagity.pipeline.graph_flow.graph_query_stream_flow`), and a
``<base>+synthesis`` mode composes the same retrieval with
:mod:`varagity.graph.answer`'s non-streaming writer — which is what the
``eval graph`` harness scores, so the harness keeps guarding the shipped
path. Unsuffixed modes keep the bake-off's engine-composed path verbatim, so
ADR-017's numbers stay reproducible.
"""

import asyncio
import json
import logging
import os
import re
import threading
import time
from collections.abc import Awaitable, Callable, Coroutine, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from varagity.config import get_settings
from varagity.graph.answer import synthesis_context, synthesis_max_chars, synthesize
from varagity.graph.base import GraphSession, register
from varagity.graph.manifest import (
    WorkdirManifest,
    load_manifest,
    load_summary,
    save_manifest,
    save_summary,
)
from varagity.graph.records import (
    BuildReport,
    GraphAnswer,
    GraphEntity,
    GraphEvidence,
    GraphExport,
    GraphExportEdge,
    GraphExportNode,
    GraphRelation,
    GraphRetrieval,
    GraphStats,
    TranscriptExcerpt,
)
from varagity.graph.render import (
    TranscriptDoc,
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
# aggregation questions get (stage-1 decision #13). The *shipped* query mode
# is a setting (`GRAPH_QUERY_MODE`, defaulting to `mix` — the acceptance
# gate's winner); this constant stays what the bake-off ran, so its numbers
# stay reproducible.
PRIMARY_MODE = "hybrid"

# The engine's accepted base modes — the vocabulary `GRAPH_QUERY_MODE` is
# validated against. `varagity/config.py` hard-codes the same tuple (it
# cannot import this module: the adapter reads settings), and a regression
# test pins the two together, exactly as the retriever and chat-engine
# registries are pinned to their validators.
QUERY_MODES = ("local", "global", "hybrid", "naive", "mix")

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

# Documents per enqueue call. LightRAG resumes from its doc-status store, so
# chunking the enqueue turns a mid-run failure into a recorded partial rather
# than a lost multi-hour index.
_INSERT_BATCH = 20

# LightRAG's enqueue stage drops a re-submitted doc_id in any status — the
# behavior an upsert wants — but records each drop as a content-less FAILED
# doc-status row under this id prefix ("File name already exists"), which its
# process pass then preserves forever. See `_purge_dup_stubs`.
_DUP_STUB_PREFIX = "dup-"

# Duck-typed stand-in for ``lightrag.base.DocStatus.FAILED``. The pinned
# JsonDocStatusStorage (see _ENV_PINS) reads only ``.value`` from status
# filters, and the real enum cannot be imported here: even a lazy import
# inside the sweep would pull the engine library into fake-driven unit runs
# and trip the import-lightness guard (only `session()` touches the library).
_DOC_STATUS_FAILED = SimpleNamespace(value="failed")

# How long `close()` waits for the session's loop thread to wind down. The
# loop is stopped, not cancelled, so the wait covers only callbacks already
# scheduled; a thread that overruns it is a daemon and dies with the process
# rather than wedging an API shutdown.
_LOOP_JOIN_TIMEOUT_S = 30.0

# Whole-graph selector for `get_knowledge_graph` (degree-ordered, capped).
_ALL_NODES_LABEL = "*"

# LightRAG packs multi-valued node/edge attributes into one string joined by
# this separator (`lightrag.constants.GRAPH_FIELD_SEP`) — `file_path` holds
# every document an entity was extracted from. Hardcoded rather than
# imported: the registry (and therefore this module) must stay free of the
# engine library, which only `session()` may touch (stage-1 decision #8).
_FIELD_SEP = "<SEP>"

# LightRAG's placeholder for material indexed without a file path. Never a
# document key, so the drill-down must not offer it as a source.
_UNKNOWN_SOURCE = "unknown_source"

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
    # The engine clamps every `get_knowledge_graph` slice to MAX_GRAPH_NODES,
    # read once at import into a dataclass field default. Its own 1000 would
    # silently re-truncate the API's raised export ceiling
    # (`varagity.api.routes.graph.MAX_EXPORT_NODES`, 5000 — a regression
    # test pins the two together).
    "MAX_GRAPH_NODES": "5000",
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
    payload: Any,
    index: Mapping[str, Sequence[str]],
    names: Mapping[str, str] | None = None,
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
        names: ``doc_key`` → display name, for labelling chunk-grain hits
            with the same thread name doc-grain hits render (``None``
            degrades to the header-parse fallback inside :func:`_excerpt`).

    Returns:
        The evidence (entities/relations; LightRAG has no community layer)
        and the retrieved transcript passages, in the engine's own relevance
        order — one per transcript document: ``mix`` mode reaches the same
        document through more than one arm, and the best-ranked hit wins
        (:func:`_dedupe_excerpts`).
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
    excerpts = _dedupe_excerpts(
        excerpt
        for item in _section(data, "chunks")
        if (excerpt := _excerpt(item, index, names or {})) is not None
    )
    return evidence, excerpts


def export_from_knowledge_graph(graph: Any) -> GraphExport:
    """Flatten LightRAG's ``KnowledgeGraph`` into the wire-ready export.

    The engine's own shape keeps everything it knows in a free-form
    ``properties`` dict and reports no degree at all, so this projects the
    two fields the view actually draws with (type, description), unpacks the
    ``file_path`` provenance the drill-down joins on
    (:func:`_doc_keys`), and counts each node's edges **within the returned
    slice** — an export is capped and degree-ordered, so a slice-local degree
    is the honest number for the picture being drawn.

    Args:
        graph: Whatever ``get_knowledge_graph`` returned (or ``None``).

    Returns:
        The export. An unknown shape degrades to an empty one rather than
        raising — the same rule the evidence normalizers follow.
    """
    nodes = _attr(graph, "nodes") or []
    edges = _attr(graph, "edges") or []
    exported_edges: list[GraphExportEdge] = []
    degrees: dict[str, int] = {}
    for edge in edges:
        source, target = _attr(edge, "source"), _attr(edge, "target")
        if not isinstance(source, str) or not isinstance(target, str):
            continue
        properties = _attr(edge, "properties")
        properties = properties if isinstance(properties, Mapping) else {}
        exported_edges.append(
            GraphExportEdge(
                id=str(_attr(edge, "id") or f"{source}-{target}"),
                source=source,
                target=target,
                label=_text(properties, "keywords", "label", "relation"),
                description=_text(properties, "description", "summary", "content"),
            )
        )
        degrees[source] = degrees.get(source, 0) + 1
        degrees[target] = degrees.get(target, 0) + 1
    exported_nodes: list[GraphExportNode] = []
    for node in nodes:
        identifier = _attr(node, "id")
        if not isinstance(identifier, str) or not identifier:
            continue
        properties = _attr(node, "properties")
        properties = properties if isinstance(properties, Mapping) else {}
        exported_nodes.append(
            GraphExportNode(
                id=identifier,
                entity_type=_text(properties, "entity_type", "type", "category"),
                description=_text(properties, "description", "summary", "content"),
                degree=degrees.get(identifier, 0),
                doc_keys=_doc_keys(properties),
            )
        )
    return GraphExport(
        nodes=exported_nodes,
        edges=exported_edges,
        truncated=bool(_attr(graph, "is_truncated")),
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
        # Lazy even though lightrag is a main dependency now (stage-1
        # decision #8): the registry stays free to import, so unit tests and
        # CI never pay for the engine — numpy included, since LightRAG's
        # EmbeddingFunc expects an array and numpy is not a declared
        # dependency of this project, only a transitive one.
        import numpy as np
        from lightrag import LightRAG, QueryParam
        from lightrag.kg.shared_storage import finalize_share_data, initialize_pipeline_status
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
        session = _LightRAGSession(
            rag, QueryParam, workdir=workdir, share_finalizer=finalize_share_data
        )
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
        workdir: Path,
        mode: str = PRIMARY_MODE,
        llm: LLMClient | None = None,
        share_finalizer: Callable[[], None] | None = None,
    ) -> None:
        """Wrap an initialized LightRAG instance and start its loop thread.

        Args:
            rag: The ``LightRAG`` instance (injected so the session's logic is
                testable against a double, with only :meth:`LightRAGEngine.session`
                touching the real library).
            query_param: LightRAG's ``QueryParam`` class, called with
                ``mode=`` / ``only_need_context=``.
            workdir: The engine's working directory — where the manifest and
                summary sidecars live. It is read here (the manifest becomes
                the provenance index, so a re-opened session can still map
                citations back to messages) and written by every method that
                changes the graph.
            mode: Primary query mode for this session.
            llm: Chat client for ``+synthesis`` answers; ``None`` builds one
                on first use, so a session that only ever runs engine-composed
                modes never opens the connection.
            share_finalizer: LightRAG's ``finalize_share_data`` (injected by
                :meth:`LightRAGEngine.session`, the only importer of the
                library). The library parks doc statuses, pipeline state, and
                update flags in **process-global** module dicts that survive
                ``finalize_storages()`` — without this reset at close, a
                close→wipe→reopen cycle (the reingest path) re-attaches to
                ghost statuses and the enqueue dedup silently vetoes the
                entire rebuild. ``None`` (tests) skips the reset.
        """
        self._rag = rag
        self._query_param = query_param
        self._workdir = workdir
        self._mode = mode
        self._llm = llm
        self._share_finalizer = share_finalizer
        self._manifest = load_manifest(workdir)
        self._index: dict[str, list[str]] = self._manifest.guid_index()
        self._names: dict[str, str] = self._manifest.thread_name_index()
        self._closed = False
        # One event loop per session, **driven by its own daemon thread**:
        # LightRAG's async primitives bind to the running loop, and running it
        # from the calling thread (`run_until_complete`) would serialize the
        # whole process behind a multi-day build. Submitting work to a loop
        # that is always running is what lets a chat query overtake an
        # in-flight extraction — the engine's own concurrency model.
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="lightrag-loop", daemon=True
        )
        self._thread.start()

    def run[T](self, coroutine: Coroutine[Any, Any, T]) -> T:
        """Run one coroutine on the session's loop and wait for it.

        Args:
            coroutine: The coroutine to drive.

        Returns:
            Whatever it returned.

        Raises:
            RuntimeError: If the session is already closed (its loop is gone,
                so the call could never complete).
        """
        if self._closed:
            coroutine.close()
            raise RuntimeError("this LightRAG session is closed")
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result()

    def build(
        self,
        batches: Sequence[MessageBatch],
        *,
        verbose: int = 0,
        prune_removed: bool = True,
    ) -> BuildReport:
        """Upsert the batches' merged messages as thread transcripts.

        The diff against the workdir manifest is what makes this an upsert
        rather than an append (stage-2 decision #9):

        * **unchanged** documents are skipped — the expensive part of a
          build is extraction, and their content already produced entities;
        * **changed** documents are deleted from the graph *first*, because
          LightRAG's enqueue stage drops a re-submitted ``doc_id`` in any
          status and would otherwise keep the stale transcript forever;
        * **removed** documents (in the manifest, absent from this render)
          are deleted when ``prune_removed`` says the render covered the
          whole corpus — and, on a bounded render, the **re-span
          casualties** among them: greedy day-span packing re-keys
          downstream documents when a bounded window grows backward, and a
          removed key whose every recorded message this render re-indexes
          under new keys is a duplicate transcript, not out-of-window
          content (:meth:`~varagity.graph.manifest.WorkdirManifest.respanned`).
          The genuinely out-of-window rest is kept.

        Enqueue and process are separate engine calls, and process re-selects
        every in-flight or failed document, so **a killed build resumes by
        calling this again** (or :meth:`resume`, which skips the render).

        Args:
            batches: Parsed source files (guid-merged before rendering).
            verbose: Validated console verbosity (0–2).
            prune_removed: Whether this render is the whole corpus. A bounded
                build (message cap, date floor) must pass ``False``: its
                render is deliberately partial, and pruning on its say-so
                would delete the rest of the archive — only its re-span
                casualties (fully re-rendered removed keys) are pruned on
                its behalf.

        Returns:
            The build report; a failed enqueue chunk or delete is recorded
            and the rest of the corpus still indexes.
        """
        messages = merge_batches(batches)
        docs = thread_transcripts(messages)
        diff = self._manifest.diff(docs)
        respanned = [] if prune_removed else self._manifest.respanned(diff.removed, docs)
        stale = [*diff.changed, *(diff.removed if prune_removed else respanned)]
        logger.info(
            "graph build: %d new, %d changed, %d unchanged, %d removed (%s)",
            len(diff.new),
            len(diff.changed),
            len(diff.unchanged),
            len(diff.removed),
            _removed_note(prune_removed, removed=len(diff.removed), respanned=len(respanned)),
        )
        started = time.perf_counter()
        undeleted = self._delete(stale)
        failures = list(undeleted.values())
        pending = set(diff.pending)
        failures.extend(self._enqueue([doc for doc in docs if doc.doc_key in pending]))
        failures.extend(self._process())
        # The manifest is rewritten even after a failed *enqueue or process*:
        # those documents were offered, and they sit in LightRAG's doc-status
        # store where the next pass retries them. A failed **delete** is the
        # exception — its old content is still in the graph, so `retain`
        # keeps its old record and the next build still sees it as stale.
        merged = self._manifest.merged(docs, prune=prune_removed, retain=undeleted.keys())
        if respanned:
            # A pruned casualty leaves the manifest with the graph; one whose
            # delete failed keeps its record (it is still in the graph) and
            # is re-spotted as removed-and-covered by the next build.
            merged = merged.without([key for key in respanned if key not in undeleted])
        self._remember(merged)
        return BuildReport(
            messages_seen=len(messages),
            wall_clock_s=time.perf_counter() - started,
            failures=failures,
        )

    def resume(self, *, verbose: int = 0) -> BuildReport:
        """Finish whatever a killed build left in flight, without re-rendering.

        LightRAG's process pass re-selects every document in a pending,
        parsing, analyzing, processing, or failed state, so this is the whole
        of resume: no corpus, no diff, no re-enqueue.

        Args:
            verbose: Validated console verbosity (0–2).

        Returns:
            The report. ``messages_seen`` is the corpus the manifest accounts
            for — nothing was handed to the engine this time, so reporting a
            hand-off count would be a lie.
        """
        started = time.perf_counter()
        failures = self._process()
        self._refresh_summary()
        return BuildReport(
            messages_seen=self._manifest.message_guid_count(),
            wall_clock_s=time.perf_counter() - started,
            failures=failures,
        )

    def document_statuses(self) -> dict[str, int]:
        """Count the engine's documents by processing status.

        Returns:
            Status name → document count (LightRAG's own vocabulary:
            ``pending``, ``processing``, ``processed``, ``failed``, …), or an
            empty mapping when the engine would not say — the progress
            sampler treats that as "no news", never as "zero documents".
        """
        try:
            statuses = self.run(self._rag.get_processing_status())
        except Exception:
            logger.warning("LightRAG document statuses unavailable", exc_info=True)
            return {}
        if not isinstance(statuses, Mapping):
            return {}
        return {str(name): int(count) for name, count in statuses.items()}

    def delete_documents(self, doc_keys: Sequence[str]) -> int:
        """Remove documents (and their derived graph elements) from the graph.

        Args:
            doc_keys: The transcript keys to delete. Unknown keys are handed
                to the engine anyway — it answers "not found" harmlessly, and
                filtering against the manifest would make a manifest drift
                permanent.

        Returns:
            How many deletes the engine accepted.

        Raises:
            RuntimeError: If the session is closed.
        """
        undeleted = self._delete(doc_keys)
        self._remember(self._manifest.without([key for key in doc_keys if key not in undeleted]))
        return len(doc_keys) - len(undeleted)

    def export(
        self,
        label: str = _ALL_NODES_LABEL,
        *,
        max_depth: int = 3,
        max_nodes: int = 5000,
    ) -> GraphExport:
        """Read a renderable slice of the graph out of the engine.

        Args:
            label: Entity name to centre the slice on; ``"*"`` takes the
                whole graph, degree-ordered.
            max_depth: Hops to walk out from ``label`` (ignored for ``"*"``).
            max_nodes: Node cap. The engine clamps it to its own
                ``max_graph_nodes`` ceiling and reports ``is_truncated`` when
                it bites.

        Returns:
            The slice, or an empty export when the graph would not answer (a
            view that cannot draw is better than a 500).
        """
        try:
            graph = self.run(
                self._rag.get_knowledge_graph(label, max_depth=max_depth, max_nodes=max_nodes)
            )
        except Exception:
            logger.warning("LightRAG graph export failed", exc_info=True)
            return GraphExport()
        return export_from_knowledge_graph(graph)

    def retrieve(
        self, question: str, *, mode: str | None = None, verbose: int = 0
    ) -> GraphRetrieval:
        """Retrieve evidence for one question without spending an answer call.

        One ``aquery_data`` call: LightRAG forces ``only_need_context``
        internally and hands back structured entities, relations, and
        chunks, which is exactly ADR-017's retrieval-only design. Both
        answer paths are built on this — :meth:`query`'s non-streaming
        synthesis and the API's streamed generation — so the ``eval graph``
        harness and a chat turn ground on the same payload.

        Args:
            question: The search query, verbatim.
            mode: LightRAG query mode, with or without the ``+synthesis``
                suffix (retrieval is the same either way — the suffix only
                names who writes the answer); ``None`` uses the session's
                primary.
            verbose: Validated console verbosity (0–2).

        Returns:
            The normalized evidence and transcript excerpts, both empty when
            retrieval failed — an unanswerable turn, never a raise.
        """
        used = mode or self._mode
        evidence, excerpts = retrieval_from_query_data(
            self._query_data(question, self._base_mode(used)), self._index, self._names
        )
        return GraphRetrieval(evidence=evidence, excerpts=excerpts, mode=used)

    def query(self, question: str, *, mode: str | None = None, verbose: int = 0) -> GraphAnswer:
        """Answer one question, with either the engine's pipeline or ours.

        A ``<base>+synthesis`` mode runs ADR-017's shipped design:
        :meth:`retrieve` (structured, no engine answer call) and a grounded
        answer written by :mod:`varagity.graph.answer` over the entities,
        relations, **and** transcript excerpts it returned.

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
        base = self._base_mode(used)
        started = time.perf_counter()
        if _split_mode(used)[1]:
            retrieval = self.retrieve(question, mode=used, verbose=verbose)
            context = synthesis_context(
                retrieval.evidence,
                retrieval.excerpts,
                max_chars=synthesis_max_chars(get_settings().LLM_CONTEXT_TOKENS),
            )
            answer_text = synthesize(self._llm_client(), question, context)
            return GraphAnswer(
                answer=answer_text,
                evidence=retrieval.evidence,
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
        """Report the graph's size, from the summary sidecar when there is one.

        The sidecar is refreshed by every build and delete, so the common
        case answers without walking a multi-megabyte graphml — which is what
        makes this cheap enough for a status poll and a metrics scrape. A
        workdir without one (a graph built before the sidecar existed, or one
        mutated behind this adapter's back) falls back to asking the graph.

        Returns:
            Entity/relation counts, or ``None`` for each if neither source
            would say. ``communities`` is always ``None``: LightRAG builds no
            community layer at all (R1), which is itself an ADR-017 datum.
        """
        summary = load_summary(self._workdir)
        if summary is not None:
            return GraphStats(
                entities=summary.entities, relations=summary.relations, communities=None
            )
        return self._graph_stats()

    def close(self) -> None:
        """Finalize LightRAG's storages, stop the loop thread, reset shared state.

        Idempotent: a second call is a no-op, so a service teardown racing a
        context manager's ``finally`` cannot raise on the closed loop.

        The shared-storage reset is the last act, once nothing can touch the
        module globals: without it the library's process-global dicts (doc
        statuses above all) outlive this session, and the next one in the
        same process re-attaches to them instead of its own on-disk state —
        which turns a reingest's close→wipe→reopen into a silently empty
        build (every re-enqueue vetoed by ghost statuses).
        """
        if self._closed:
            return
        try:
            self.run(self._rag.finalize_storages())
        except Exception:
            logger.warning("LightRAG teardown failed", exc_info=True)
        finally:
            self._closed = True
            self._drain_loop_tasks()
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=_LOOP_JOIN_TIMEOUT_S)
            if self._thread.is_alive():
                # Daemon thread: it dies with the process. Closing the loop
                # out from under it would be the noisier failure.
                logger.warning("LightRAG loop thread did not stop within %ss", _LOOP_JOIN_TIMEOUT_S)
            else:
                self._loop.close()
            if self._share_finalizer is not None:
                try:
                    self._share_finalizer()
                except Exception:  # teardown hygiene never raises
                    logger.warning("LightRAG shared-storage reset failed", exc_info=True)

    def _drain_loop_tasks(self) -> None:
        """Cancel LightRAG's own background tasks before the loop goes away.

        ``finalize_storages`` flushes data but leaves the library's long-lived
        workers (its priority-queue rate limiters, its health check) parked on
        the loop. Stopping and closing the loop under them strands each one
        mid-await: asyncio logs "Task was destroyed but it is pending!" and
        their cleanup throws "Event loop is closed" into the unraisable hook
        on every shutdown — live noise, not a data problem, but ERROR-grade
        spew in the API's logs. Cancelling them while the loop is still
        running lets them unwind properly; one that will not die within the
        join timeout is logged and abandoned to the daemon thread.
        """

        async def drain() -> None:
            """Cancel every task on the loop but ourselves, then let them end."""
            tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        try:
            asyncio.run_coroutine_threadsafe(drain(), self._loop).result(
                timeout=_LOOP_JOIN_TIMEOUT_S
            )
        except Exception:
            logger.warning("LightRAG loop drain failed", exc_info=True)

    def _graph_stats(self) -> GraphStats:
        """Count the graph's nodes and edges from LightRAG's graph storage.

        Returns:
            Entity/relation counts, or ``None`` for each if the storage would
            not say.
        """
        entities = relations = None
        try:
            graph = self.run(
                self._rag.chunk_entity_relation_graph.get_knowledge_graph(_ALL_NODES_LABEL)
            )
            entities = len(graph.nodes)
            relations = len(graph.edges)
        except Exception:  # an engine that won't report is reported as unknown
            logger.warning("LightRAG graph stats unavailable", exc_info=True)
        return GraphStats(entities=entities, relations=relations, communities=None)

    def _enqueue(self, docs: Sequence[TranscriptDoc]) -> list[str]:
        """Hand documents to LightRAG's durable pending queue, in chunks.

        Args:
            docs: The transcripts to (re-)index. Each is enqueued under its
                ``doc_key`` as both document id and ``file_path`` — the id
                makes the upsert recognizable, the path is what a retrieved
                chunk carries back as provenance.

        Returns:
            Human-readable failures, one per failed chunk.
        """
        failures: list[str] = []
        for start in range(0, len(docs), _INSERT_BATCH):
            chunk = docs[start : start + _INSERT_BATCH]
            try:
                self.run(
                    self._rag.apipeline_enqueue_documents(
                        [doc.text for doc in chunk],
                        ids=[doc.doc_key for doc in chunk],
                        file_paths=[doc.doc_key for doc in chunk],
                    )
                )
            except Exception as exc:  # a failed chunk is data, not the end of the run
                logger.warning("LightRAG enqueue failed for %d document(s)", len(chunk))
                failures.append(f"enqueue {chunk[0].doc_key}…(+{len(chunk) - 1}): {exc!r}")
        return failures

    def _process(self) -> list[str]:
        """Run LightRAG's extraction pass over everything it has pending.

        The ``dup-*`` sweep runs first, so dedup receipts born from this
        build's enqueue (and any residue an earlier run left) are gone
        before the pass — and before anything samples document statuses.

        Returns:
            A single-entry failure list if the pass raised, else empty. The
            pass is all-or-nothing from here; per-document failures live in
            the engine's doc-status store, where the next pass retries them.
        """
        self._purge_dup_stubs()
        try:
            self.run(self._rag.apipeline_process_enqueue_documents())
        except Exception as exc:
            logger.warning("LightRAG document processing failed", exc_info=True)
            return [f"process: {exc!r}"]
        return []

    def _purge_dup_stubs(self) -> None:
        """Sweep LightRAG's ``dup-*`` dedup receipts out of the doc-status store.

        The enqueue stage drops a re-submitted ``doc_id`` in any status —
        exactly what the manifest-diffed upsert wants — but 1.5.4 records
        each drop as a content-less ``dup-``-prefixed FAILED row, and its
        process pass excludes those stubs from work without ever deleting
        them ("preserved for manual review"). Left alone they accumulate
        one per re-offered document per resumed build, and every status
        surface reads ``failed > 0`` on a healthy graph. The manifest is
        this adapter's dedup ledger, so the receipts carry nothing: purging
        keeps :meth:`document_statuses` honest. Real failures keep their
        rows — only the ``dup-`` prefix is swept, and hygiene never fails a
        build (any error is logged and swallowed).
        """

        async def sweep() -> list[str]:
            store = self._rag.doc_status
            failed = await store.get_docs_by_statuses([_DOC_STATUS_FAILED])
            stubs = [doc_id for doc_id in failed if doc_id.startswith(_DUP_STUB_PREFIX)]
            if stubs:
                await store.delete(stubs)
                await store.index_done_callback()  # delete defers its disk flush
            return stubs

        try:
            stubs = self.run(sweep())
        except Exception:  # hygiene must never fail a build
            logger.warning("could not purge the engine's dup-* doc-status stubs", exc_info=True)
            return
        if stubs:
            logger.info(
                "purged %d dup-* dedup receipt(s) from the engine's doc-status store", len(stubs)
            )

    def _delete(self, doc_keys: Sequence[str]) -> dict[str, str]:
        """Delete documents from the graph one at a time.

        One call per document on purpose: the engine rebuilds the entities a
        deleted document partly supported, and a failure on one key must not
        strand the rest of a re-index.

        Args:
            doc_keys: The transcript keys to remove.

        Returns:
            ``doc_key`` → failure text, for the keys that would not delete.
            The caller needs the *keys*, not just the messages: a document
            still in the graph must keep its old manifest record, or the
            next build will think it is up to date.
        """
        failures: dict[str, str] = {}
        for key in doc_keys:
            try:
                self.run(self._rag.adelete_by_doc_id(key))
            except Exception as exc:
                logger.warning("LightRAG delete failed for %s", key, exc_info=True)
                failures[key] = f"delete {key}: {exc!r}"
        return failures

    def _remember(self, manifest: WorkdirManifest) -> None:
        """Adopt a new manifest: persist it, re-derive the provenance indexes.

        Args:
            manifest: The manifest describing what the graph now holds.
        """
        self._manifest = manifest
        self._index = manifest.guid_index()
        self._names = manifest.thread_name_index()
        save_manifest(self._workdir, manifest)
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        """Rewrite the summary sidecar from a fresh graph walk.

        Called only after a write (build, resume, delete) — never on a read
        path, which is the whole point of having a sidecar.
        """
        stats = self._graph_stats()
        save_summary(
            self._workdir,
            self._manifest,
            entities=stats.entities,
            relations=stats.relations,
        )

    def _base_mode(self, mode: str) -> str:
        """Resolve the engine mode a query string actually retrieves with.

        Args:
            mode: The caller's mode, optionally ``+synthesis``-suffixed.

        Returns:
            The base engine mode: the caller's when it named one, else the
            session's primary (a bare ``"synthesis"`` names none).
        """
        base = _split_mode(mode)[0]
        return base or _split_mode(self._mode)[0] or PRIMARY_MODE

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


def _removed_note(prune_removed: bool, *, removed: int, respanned: int) -> str:
    """Phrase the build log's removed-keys disposition.

    Args:
        prune_removed: Whether the render covered the whole corpus.
        removed: How many manifest keys the render did not mention.
        respanned: How many of them are re-span casualties (bounded runs).

    Returns:
        The parenthetical for the build-diff log line.
    """
    if prune_removed:
        return "pruning removed"
    if respanned:
        return f"bounded render — pruning {respanned} re-spanned, keeping {removed - respanned}"
    return "bounded render — keeping removed"


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


def _attr(item: Any, name: str) -> Any:
    """Read one field from a payload that may be a model or a mapping.

    ``get_knowledge_graph`` returns pydantic models, but a JSON round-trip
    (a cached export, a test fixture) returns dicts of the same shape.

    Args:
        item: The record, or ``None``.
        name: Field name.

    Returns:
        The field's value, or ``None`` when the record has no such field.
    """
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)


def _doc_keys(properties: Mapping[str, Any]) -> list[str]:
    """Read the transcript documents one graph node was extracted from.

    LightRAG merges an entity's provenance into a single ``file_path``
    string joined by :data:`_FIELD_SEP`, and writes its own
    ``"unknown_source"`` placeholder for material that arrived without one.
    Both are normalized away here so the drill-down gets real document keys
    or nothing — a placeholder rendered as a source day would be a fiction.

    Reads the raw field rather than :func:`_text`, deliberately: this is the
    one consumer for which the separator *is* the data, and `_text`
    normalizes it away.

    Args:
        properties: The node's free-form property mapping.

    Returns:
        The distinct document keys, first-seen order.
    """
    packed = properties.get("file_path")
    if not isinstance(packed, str) or not packed.strip():
        return []
    keys: list[str] = []
    for candidate in packed.split(_FIELD_SEP):
        key = candidate.strip()
        if key and key != _UNKNOWN_SOURCE and key not in keys:
            keys.append(key)
    return keys


def _text(item: Mapping[str, Any], *names: str) -> str | None:
    """Read the first present non-empty string field from a payload item.

    The engine merges every re-extraction of an entity/relation into one
    stored string joined by :data:`_FIELD_SEP` — an internal storage detail
    that must not reach anything a person (or the synthesis prompt) reads,
    so it is normalized to a single space here, at the one seam every
    description passes through.

    Args:
        item: One entity/relation record.
        *names: Candidate field names, most likely first.

    Returns:
        The trimmed, separator-normalized value, or ``None`` when no
        candidate holds text.
    """
    for name in names:
        value = item.get(name)
        if isinstance(value, str) and value.strip():
            text = value.strip()
            if _FIELD_SEP in text:
                text = " ".join(part.strip() for part in text.split(_FIELD_SEP) if part.strip())
            return text
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
    item: Mapping[str, Any],
    index: Mapping[str, Sequence[str]],
    names: Mapping[str, str],
) -> TranscriptExcerpt | None:
    """Map one retrieved chunk into a transcript excerpt.

    The adapter inserts every document under its ``doc_key`` as both id and
    ``file_path``, so a chunk's ``file_path`` *is* the corpus join key — and
    provenance resolves through the same tolerant walk the rest of the
    adapter uses (LightRAG joins multiple paths with its own separator).

    The thread label resolves manifest-first: only a doc-grain hit carries
    the transcript header in its text, so parsing alone would label a
    chunk-grain hit of the *same document* with the raw thread id — two
    names for one citation target. The manifest knows the display name for
    both grains; the header parse and the id stay as fallbacks for a
    workdir indexed before the manifest recorded names.

    Args:
        item: One ``chunks`` record.
        index: ``doc_key`` → message guids.
        names: ``doc_key`` → display name
            (:meth:`~varagity.graph.manifest.WorkdirManifest.thread_name_index`).

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
        thread_name=names.get(doc_key) or (header.group(1) if header else thread_id),
        span=span,
        text=text,
        message_guids=guids_in_payload(doc_key, index),
    )


def _dedupe_excerpts(excerpts: Iterable[TranscriptExcerpt]) -> list[TranscriptExcerpt]:
    """Collapse a retrieval onto one excerpt per transcript document.

    ``mix`` mode reaches the same document through more than one arm
    (vector chunks plus entity/relation text-units), and every consumer
    downstream — the answer prompt, the SSE ``retrieval`` event, the
    persisted ``message_sources`` rows — wants each document once, under
    one label. Hits arrive best-first, so the first per ``doc_key`` wins;
    a dropped duplicate contributes only its thread name, and only when
    the kept hit could not resolve one past the thread id (a manifest-less
    workdir, where a chunk-grain hit has no header to parse).

    Args:
        excerpts: Mapped excerpts, in the engine's relevance order.

    Returns:
        One excerpt per document, first-hit order.
    """
    kept: dict[str, TranscriptExcerpt] = {}
    for excerpt in excerpts:
        seen = kept.get(excerpt.doc_key)
        if seen is None:
            kept[excerpt.doc_key] = excerpt
        elif (
            seen.thread_name == seen.doc_key.partition("::")[0]
            and excerpt.thread_name != seen.thread_name
        ):
            seen.thread_name = excerpt.thread_name
    return list(kept.values())
