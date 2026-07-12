"""The HTTP wire contract — pydantic request/response models (spec_v2 §4.1).

Every request and response body is a model here; FastAPI renders them into
the OpenAPI schema the web app's TypeScript types are generated from
(``openapi-typescript``), so the contract cannot drift by hand-editing.
SSE event payloads (spec_v2 §4.3) are models too — one per event type —
because they cross the same wire.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from varagity.stores.records import RetrievedChunk


class ErrorBody(BaseModel):
    """The machine-readable error inside the envelope (spec_v2 §4.1).

    Attributes:
        code: Stable, machine-readable code (e.g. ``"es_unreachable"``) the
            GUI maps to an actionable banner.
        message: Human-readable detail.
    """

    code: str
    message: str


class ErrorResponse(BaseModel):
    """The structured error envelope every non-2xx response carries.

    Attributes:
        error: The error body.
    """

    error: ErrorBody


class ChatOverrides(BaseModel):
    """Per-request overrides of query-time settings (spec_v2 §4.2).

    Phase 2 honors the two knobs the thin client needs; the persisted
    runtime-override layer (spec_v2 §4.7) lands with its GUI in Phase 8.
    Unknown fields are rejected so a typo'd override fails loudly instead
    of silently running with defaults.

    Attributes:
        retrieval_method: Registry name of the retrieval method for this
            question only (``semantic`` | ``bm25`` | ``hybrid`` |
            ``reranked``).
        top_k: Number of chunks retrieved for this question only.
    """

    model_config = ConfigDict(extra="forbid")

    retrieval_method: str | None = None
    top_k: int | None = Field(default=None, ge=1)


class ChatRequest(BaseModel):
    """Body of ``POST /api/chat`` (spec_v2 §4.2).

    Attributes:
        query: The user's question.
        conversation_id: Existing conversation to append the turn to; a new
            conversation is created when omitted.
        overrides: Optional per-request setting overrides.
    """

    query: str = Field(min_length=1)
    conversation_id: str | None = None
    overrides: ChatOverrides | None = None


class RetrievalEvent(BaseModel):
    """Payload of the SSE ``retrieval`` event — the provenance panel's data.

    Emitted once retrieval (+rerank) completes, **before** any answer token
    (spec_v2 §4.3): the browser gets the evidence before the prose.

    Attributes:
        chunks: The retrieved chunks, best first, each carrying its full
            metadata record and (when the method fills it) the
            :class:`~varagity.stores.records.RetrievalTrace`.
        method: The retrieval method that produced them.
        top_k: Chunks requested from the retriever.
        reranked_to: ``RERANK_TOP_N`` when the ``reranked`` method narrowed
            the list; ``None`` otherwise.
    """

    chunks: list[RetrievedChunk]
    method: str
    top_k: int
    reranked_to: int | None = None


class DeltaEvent(BaseModel):
    """Payload of the SSE ``reasoning`` and ``token`` events.

    Attributes:
        delta: The next text fragment, in stream order.
    """

    delta: str


class UsageInfo(BaseModel):
    """Token usage and per-stage latency reported by the ``done`` event.

    Attributes:
        prompt_tokens: Server-reported prompt tokens (``None`` when the
            model server reports no usage).
        completion_tokens: Server-reported completion tokens (``None`` when
            unreported).
        latency_ms: Wall-clock milliseconds per stage: ``retrieval``,
            ``generation``, ``total``.
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: dict[str, int]


class DoneEvent(BaseModel):
    """Payload of the terminal SSE ``done`` event (spec_v2 §4.3).

    Attributes:
        message_id: Persisted assistant message id.
        conversation_id: The conversation the turn was persisted into (the
            one just created, when the request named none).
        answer: The full, ``<think>``-stripped answer (authoritative — the
            streamed deltas are best-effort display).
        usage: Token usage and per-stage latency.
    """

    message_id: str
    conversation_id: str
    answer: str
    usage: UsageInfo


class ErrorEvent(BaseModel):
    """Payload of the in-band SSE ``error`` event.

    Emitted when the pipeline fails after the stream already opened (the
    200 headers are gone — status can't change; spec_v2 plan decision #7).

    Attributes:
        code: Stable, machine-readable code.
        message: Human-readable detail.
    """

    code: str
    message: str


class ConversationCreateRequest(BaseModel):
    """Body of ``POST /api/conversations``.

    Attributes:
        title: Optional explicit title; defaults to the placeholder the
            first chat turn auto-replaces.
    """

    title: str | None = Field(default=None, min_length=1, max_length=200)


class ConversationSummaryOut(BaseModel):
    """One conversation in the ``GET /api/conversations`` list.

    Attributes:
        conversation_id: The app-generated id.
        title: Current title.
        created_at: Creation timestamp.
        updated_at: Last-turn timestamp (list ordering key).
        message_count: Number of persisted messages.
    """

    conversation_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int


class MessageSourceOut(BaseModel):
    """One snapshotted evidence row of a persisted assistant turn.

    Attributes:
        rank: Final rank in the answer's evidence (1-based).
        chunk_id: Soft reference to the producing chunk.
        trace: The spec_v2 §9.1 snapshot: score, content, context, source
            provenance, and the serialized retrieval trace.
    """

    rank: int
    chunk_id: str
    trace: dict[str, Any]


class MessageOut(BaseModel):
    """One persisted message in a transcript.

    Attributes:
        message_id: The app-generated id.
        role: ``"user"`` or ``"assistant"``.
        content: The question or the generated answer.
        created_at: Persistence timestamp.
        retrieval_method: Retrieval method of an assistant turn.
        latency_ms: Per-stage timings of an assistant turn.
        reasoning: Captured ``<think>`` stream, if any.
        sources: The turn's snapshotted evidence, rank order.
    """

    message_id: str
    role: str
    content: str
    created_at: datetime
    retrieval_method: str | None = None
    latency_ms: dict[str, Any] | None = None
    reasoning: str | None = None
    sources: list[MessageSourceOut] = []


class ConversationDetailOut(BaseModel):
    """Response of ``GET /api/conversations/{id}`` — the full transcript.

    Attributes:
        conversation_id: The app-generated id.
        title: Current title.
        created_at: Creation timestamp.
        updated_at: Last-turn timestamp.
        messages: All messages, oldest first, each with its sources.
    """

    conversation_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageOut]


class ServiceHealth(BaseModel):
    """Reachability of one backing service (spec_v2 §4.2).

    Attributes:
        ok: Whether the service answered its probe.
        detail: Failure detail when it didn't (``None`` when healthy).
    """

    ok: bool
    detail: str | None = None


class HealthResponse(BaseModel):
    """Response of ``GET /api/health``.

    The endpoint answers ``200`` whenever the API process itself is alive —
    per-dependency state lives in the body, so a flapping backing service
    doesn't mark the ``api`` container unhealthy.

    Attributes:
        ok: ``True`` when every dependency probe succeeded.
        services: Probe result per service name (``llamacpp``,
            ``infinity``, ``postgres``, ``elasticsearch``, ``prefect``).
    """

    ok: bool
    services: dict[str, ServiceHealth]


class NumericRange(BaseModel):
    """Valid range of one numeric setting, for GUI control generation.

    Attributes:
        min: Inclusive lower bound (``None`` = unbounded).
        max: Inclusive upper bound (``None`` = unbounded).
    """

    min: float | None = None
    max: float | None = None


class ConfigResponse(BaseModel):
    """Response of ``GET /api/config`` — static capabilities (spec_v2 §4.2).

    The GUI builds its settings controls from this, so a newly registered
    implementation (one file + one import) appears in the UI automatically.

    Attributes:
        retrievers: Registered retrieval method names.
        chunkers: Registered chunking strategy names.
        ocr_engines: Available OCR engine names.
        model_types: Valid ``get_model`` types.
        ranges: Valid ranges for the numeric query-time knobs, keyed by
            setting name (lowercase).
    """

    retrievers: list[str]
    chunkers: list[str]
    ocr_engines: list[str]
    model_types: list[str]
    ranges: dict[str, NumericRange]
