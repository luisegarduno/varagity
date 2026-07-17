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

# The JSON scalar types a setting value can take on the wire (spec_v2 §4.7).
# bool precedes int so pydantic's smart union keeps True/False boolean.
SettingValue = bool | int | float | str


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


class StatsEvent(BaseModel):
    """Payload of the SSE ``stats`` event — live decode throughput.

    Emitted while generation runs, interleaved with the ``reasoning``/
    ``token`` deltas and throttled server-side (the model server reports
    these counters on *every* chunk; a frame per token would be noise).
    Purely additive to the protocol: the event only exists when the model
    server reports its own timings — llama.cpp does, so the readout is
    llama.cpp-only by construction rather than by configuration.

    Attributes:
        tokens_per_second: Decode throughput so far, averaged over the
            whole generation (cumulative, not instantaneous — it settles
            rather than jitters).
        completion_tokens: Tokens decoded so far, the denominator behind
            that average.
    """

    tokens_per_second: float
    completion_tokens: int


class UsageInfo(BaseModel):
    """Token usage and per-stage latency reported by the ``done`` event.

    Attributes:
        prompt_tokens: Server-reported prompt tokens (``None`` when the
            model server reports no usage).
        completion_tokens: Server-reported completion tokens (``None`` when
            unreported).
        latency_ms: Wall-clock milliseconds per stage: ``retrieval``,
            ``generation``, ``total``.
        tokens_per_second: Final decode throughput as the model server
            measured it, or ``None`` when it reports no timings. Distinct
            from ``completion_tokens / latency_ms["generation"]``, which
            would smear queueing and prompt eval into the rate.
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: dict[str, int]
    tokens_per_second: float | None = None


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
        llm_model_types: The chat-capable subset of ``model_types`` — the
            ``CHAT_MODEL_TYPE`` vocabulary the composer quick-toggle offers.
        ranges: Valid ranges for the numeric query-time knobs, keyed by
            setting name (lowercase).
        upload_max_mb: Effective per-file upload cap (``UPLOAD_MAX_MB``) —
            the dropzone validates against it client-side.
        allowed_extensions: Effective ingestable extensions
            (``ALLOWED_EXTENSIONS``, normalized with leading dots), sorted.
        preview_enabled: Whether the page-preview endpoints are on
            (``PREVIEW_ENABLED``, read-only here — the knob is env-only) —
            off, the GUI skips preview eligibility entirely.
    """

    retrievers: list[str]
    chunkers: list[str]
    ocr_engines: list[str]
    model_types: list[str]
    llm_model_types: list[str]
    ranges: dict[str, NumericRange]
    upload_max_mb: int
    allowed_extensions: list[str]
    preview_enabled: bool


class SettingOut(BaseModel):
    """One effective setting in the ``GET /api/settings`` catalog (spec_v2 §4.7).

    Attributes:
        name: The ``Settings`` field name (e.g. ``"RETRIEVAL_METHOD"``).
        value: The effective value (env defaults merged with any override).
        group: Drawer group — ``retrieval`` | ``generation`` | ``ingestion``.
        overridden: Whether a persisted runtime override is in effect (the
            drawer shows a reset affordance).
        reingest_affecting: Whether changing it marks the corpus stale (it
            doesn't change content hashes — the surfaced v1 footgun).
        choices: Valid values for enum-like settings (registry-derived, so a
            new implementation appears automatically); ``None`` for numeric
            and free-form settings (ranges live in ``GET /api/config``).
    """

    name: str
    value: SettingValue
    group: str
    overridden: bool
    reingest_affecting: bool
    choices: list[str] | None = None


class SettingsResponse(BaseModel):
    """Response of ``GET /api/settings`` and ``PATCH /api/settings``.

    Attributes:
        settings: The full overridable catalog with effective values.
        corpus_stale: Whether a reingest-affecting setting changed since the
            corpus was last (re)ingested — the "Re-ingest to apply" banner.
    """

    settings: list[SettingOut]
    corpus_stale: bool


class SettingsPatchRequest(BaseModel):
    """Body of ``PATCH /api/settings``.

    Attributes:
        overrides: Setting name → new value, or ``None`` to clear that
            override (reverting to the env value). Linked settings (the
            fusion weight pair) must be patched together — validation runs
            on the merged whole.
    """

    model_config = ConfigDict(extra="forbid")

    overrides: dict[str, SettingValue | None] = Field(min_length=1)


class DocumentOut(BaseModel):
    """One corpus document in the ``GET /api/documents`` list (spec_v2 §4.2).

    Attributes:
        doc_id: The document's stable id.
        file_name: Base name of the source file.
        source: Absolute file path recorded at ingest time.
        file_type: File extension without the dot (``pdf``, ``docx``, …).
        content_hash: sha256 of the source file's bytes at ingest time.
        n_chunks: Chunks ingested (``0`` = no extractable text).
        ingested_at: When the document (last) landed in the stores.
        extraction_mix: Chunk count per extraction method (``text`` /
            ``ocr_fallback``).
    """

    doc_id: str
    file_name: str
    source: str
    file_type: str
    content_hash: str
    n_chunks: int
    ingested_at: datetime
    extraction_mix: dict[str, int]


class PreviewRect(BaseModel):
    """One highlight rectangle, normalized to the page (``[0, 1]``, top-left origin).

    Y-flipped from PDF coordinates server-side, so the client positions
    overlay divs with bare percentages and no coordinate math.

    Attributes:
        x0: Left edge.
        y0: Top edge.
        x1: Right edge (``> x0``).
        y1: Bottom edge (``> y0``).
    """

    x0: float
    y0: float
    x1: float
    y1: float


class PreviewLocateRequest(BaseModel):
    """Body of ``POST /api/documents/{doc_id}/preview/locate`` (ADR-010).

    Attributes:
        text: The chunk content to locate — both wire shapes (the live
            ``retrieval`` event and persisted ``message_sources`` snapshots)
            already deliver it to the client, so history previews work
            without any migration.
    """

    text: str = Field(min_length=1, max_length=20_000)


class PreviewLocateResponse(BaseModel):
    """Where a chunk's text lives in its source document (ADR-010).

    ``available=False`` + ``reason`` covers every degradable condition
    (``preview_disabled`` | ``unsupported_type`` | ``file_missing`` |
    ``file_changed`` | ``conversion_unavailable`` | ``conversion_failed`` |
    ``no_match``) — the GUI falls back to the full-text view on any of
    them, never a dead panel.

    Attributes:
        available: Whether a page was located (the fields below are set).
        reason: The degradable condition when ``available`` is false.
        page: Best-matching page, 1-based.
        page_count: Total pages in the document (also set on ``no_match``).
        rects: Highlight rectangles for the chunk's text on that page.
        coverage: The winning page's containment score in ``[0, 1]`` (also
            set on ``no_match`` — the score that stayed below the floor).
    """

    available: bool
    reason: str | None = None
    page: int | None = None
    page_count: int | None = None
    rects: list[PreviewRect] = []
    coverage: float | None = None


class UploadedFileOut(BaseModel):
    """Outcome for one file of a ``POST /api/documents`` upload.

    Attributes:
        file_name: The stored (sanitized) file name (rejections echo the
            client-supplied name/path they rejected).
        size_bytes: Bytes written (``0`` when rejected).
        stored: Whether the file landed in ``DOCS_PATH``.
        replaced: Whether an existing file at the same target was
            overwritten (a re-upload; the next ingest re-processes it under
            a new hash).
        reason: Rejection reason when ``stored`` is false
            (``extension_not_allowed`` | ``file_too_large`` |
            ``invalid_filename`` | ``invalid_path`` | ``path_too_deep`` |
            ``write_failed`` — the last is a server-side problem, escalated
            to a structured ``500`` when no file in the batch landed).
        relative_path: The stored path relative to ``DOCS_PATH`` when the
            upload declared one (folder uploads, spec_v3 §5.2); ``None``
            for flat uploads and rejections.
    """

    file_name: str
    size_bytes: int
    stored: bool
    replaced: bool = False
    reason: str | None = None
    relative_path: str | None = None


class UploadResponse(BaseModel):
    """Response of ``POST /api/documents``.

    Attributes:
        files: Per-file outcomes, in upload order.
    """

    files: list[UploadedFileOut]


class DocumentDeleteResponse(BaseModel):
    """Response of ``DELETE /api/documents/{doc_id}`` (spec_v2 §4.2).

    Also one entry of a bulk delete's ``deleted`` list.

    Attributes:
        doc_id: The deleted document.
        chunks_deleted: Chunks removed (the pgvector count; the
            Elasticsearch ``delete_by_query`` removes the same identity-
            addressed set).
        file_removed: Whether the source file was also deleted from
            ``DOCS_PATH`` (requested via ``?remove_file=true`` and only
            honored inside the corpus directory).
    """

    doc_id: str
    chunks_deleted: int
    file_removed: bool


class DocumentBulkDeleteRequest(BaseModel):
    """Body of ``POST /api/documents/delete`` (spec_v2 §4.2).

    Attributes:
        doc_ids: The documents to remove, at least one. Duplicates collapse
            and unknown ids are reported rather than fatal, so the corpus
            table's selection can be posted as-is.
    """

    model_config = ConfigDict(extra="forbid")

    doc_ids: list[str] = Field(min_length=1)


class DocumentBulkDeleteResponse(BaseModel):
    """Response of ``POST /api/documents/delete`` (spec_v2 §4.2).

    Attributes:
        deleted: Per-document outcomes, in requested order (duplicates
            collapsed) — the same shape a single delete returns.
        not_found: Requested ids that no longer had a ``documents`` row.
            Reported, never fatal: a concurrent delete (another tab, a
            reingest) must not fail the rest of the batch.
    """

    deleted: list[DocumentDeleteResponse]
    not_found: list[str]


class IngestStartRequest(BaseModel):
    """Body of ``POST /api/ingest``.

    Attributes:
        reingest: Delete each discovered document's previous ingest and
            re-process it (required after reingest-affecting setting
            changes — content hashes don't change, so unchanged files are
            otherwise skipped).
    """

    reingest: bool = False


class IngestSummaryOut(BaseModel):
    """The ingest run counters (mirrors the loader's ``IngestSummary``).

    Attributes:
        discovered: Files found in the corpus buckets.
        ingested: Files parsed, chunked, embedded, and stored this run.
        skipped: Unchanged files skipped via the idempotency check.
        no_text: Files with no extractable text.
        unsupported: Files whose bucket has no registered parser.
        failed: Files that raised during ingestion (run continued).
        chunks: Total chunks stored this run.
    """

    discovered: int
    ingested: int
    skipped: int
    no_text: int
    unsupported: int
    failed: int
    chunks: int


class IngestRunOut(BaseModel):
    """One ingest run's state (``POST /api/ingest`` + the status stream).

    Attributes:
        run_id: The run handle.
        state: ``running`` | ``completed`` | ``failed``.
        reingest: Whether the run re-processes unchanged files.
        started_at: When the run started.
        finished_at: When it reached a terminal state (``None`` while
            running).
        summary: The final counters (terminal states only; ``failed`` runs
            may carry ``None`` when the flow died before summarizing).
        error: The flow-level failure (``failed`` only; per-file failures
            ride in ``summary.failed`` and the ``log`` events instead).
    """

    run_id: str
    state: str
    reingest: bool
    started_at: datetime
    finished_at: datetime | None = None
    summary: IngestSummaryOut | None = None
    error: str | None = None


class IngestStatusEvent(BaseModel):
    """Payload of the ingest-status SSE ``status`` event.

    The stream's first frame (a snapshot on connect) and its last (the
    terminal state). ``run=None`` means no ingest has run in this API
    process — the stream closes immediately after.

    Attributes:
        run: The current (or last) run, if any.
    """

    run: IngestRunOut | None = None


class IngestProgressEvent(BaseModel):
    """Payload of the ingest-status SSE ``progress`` event.

    One frame per pipeline stage transition, mirroring the CLI's ``rich``
    display: ``discover`` (with ``total`` files), then per file ``parse`` →
    ``chunk`` → ``contextualize`` (with per-chunk ``current``/``total``
    ticks) → ``embed`` → ``store``, and a ``file_done`` frame carrying the
    file's outcome plus the run-level progress counters.

    Attributes:
        stage: ``discover`` | ``parse`` | ``chunk`` | ``contextualize`` |
            ``embed`` | ``store`` | ``file_done``.
        file: The file being processed (``None`` for ``discover``).
        outcome: ``file_done`` only — ``ingested`` | ``skipped`` |
            ``no_text`` | ``failed``.
        current: Intra-stage progress (contextualized chunks so far).
        total: Stage denominator (files for ``discover``, chunks for
            ``chunk``/``contextualize``, texts for ``embed``).
        files_done: Files finished so far (``file_done`` only).
        files_total: Files discovered (``file_done`` only).
    """

    stage: str
    file: str | None = None
    outcome: str | None = None
    current: int | None = None
    total: int | None = None
    files_done: int | None = None
    files_total: int | None = None


class IngestLogEvent(BaseModel):
    """Payload of the ingest-status SSE ``log`` event.

    Relayed ``varagity.ingest`` log records (the per-file outcome lines the
    terminal shows: skipped-unchanged, no-text, failure tracebacks' heads),
    so the browser progress view mirrors the CLI run.

    Attributes:
        level: The log level name (``INFO`` | ``WARNING`` | ``ERROR``).
        message: The formatted log message.
    """

    level: str
    message: str
