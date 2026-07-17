/**
 * Typed client for the Varagity HTTP API (spec_v2 §4).
 *
 * Every wire shape is re-exported from the generated `lib/types.ts`
 * (`bun run gen:types` against the live `/openapi.json`) — nothing here is
 * hand-maintained. `streamChat` speaks the POST-SSE chat protocol:
 * `fetch()` → `response.body` → `eventsource-parser`, because the native
 * `EventSource` is GET-only and cannot carry the JSON request body.
 */
import { EventSourceParserStream } from "eventsource-parser/stream";

import type { components } from "@/lib/types";

type Schemas = components["schemas"];

export type ChatRequest = Schemas["ChatRequest"];
export type ChatOverrides = Schemas["ChatOverrides"];
export type ConversationSummary = Schemas["ConversationSummaryOut"];
export type ConversationDetail = Schemas["ConversationDetailOut"];
export type ChatMessage = Schemas["MessageOut"];
export type RetrievalEvent = Schemas["RetrievalEvent"];
export type RetrievedChunk = Schemas["RetrievedChunk"];
export type RetrievalTrace = Schemas["RetrievalTrace"];
export type DeltaEvent = Schemas["DeltaEvent"];
export type StatsEvent = Schemas["StatsEvent"];
export type DoneEvent = Schemas["DoneEvent"];
export type ChatErrorEvent = Schemas["ErrorEvent"];
export type ErrorResponse = Schemas["ErrorResponse"];
export type HealthResponse = Schemas["HealthResponse"];
export type ConfigResponse = Schemas["ConfigResponse"];
export type SettingOut = Schemas["SettingOut"];
export type SettingsResponse = Schemas["SettingsResponse"];
export type DocumentOut = Schemas["DocumentOut"];
export type PreviewRect = Schemas["PreviewRect"];
export type PreviewLocateRequest = Schemas["PreviewLocateRequest"];
export type PreviewLocateResponse = Schemas["PreviewLocateResponse"];
export type UploadResponse = Schemas["UploadResponse"];
export type UploadedFile = Schemas["UploadedFileOut"];
export type DocumentDeleteResponse = Schemas["DocumentDeleteResponse"];
export type DocumentBulkDeleteResponse = Schemas["DocumentBulkDeleteResponse"];
export type IngestRun = Schemas["IngestRunOut"];
export type IngestSummary = Schemas["IngestSummaryOut"];
export type IngestStatusEvent = Schemas["IngestStatusEvent"];
export type IngestProgressEvent = Schemas["IngestProgressEvent"];
export type IngestLogEvent = Schemas["IngestLogEvent"];

/** A JSON scalar a setting value can take on the wire (spec_v2 §4.7). */
export type SettingValue = boolean | number | string;

/**
 * Browser-reachable API origin. Inlined at build time (`NEXT_PUBLIC_*`),
 * so the compose build passes it as a build arg.
 */
export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/** A non-2xx API response, carrying the structured `{code, message}` envelope. */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string;

  constructor(status: number, code: string, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

/** Parse a failed response's error envelope into an {@link ApiError}. */
async function toApiError(response: Response): Promise<ApiError> {
  let code = `http_${response.status}`;
  let message = response.statusText || `Request failed (${response.status})`;
  try {
    const body = (await response.json()) as ErrorResponse;
    if (body?.error?.code) {
      code = body.error.code;
      message = body.error.message;
    }
  } catch {
    // Non-JSON failure body (proxy error page, …) — keep the status text.
  }
  return new ApiError(response.status, code, message);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "content-type": "application/json", ...init?.headers },
  });
  if (!response.ok) throw await toApiError(response);
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

/** List conversations, most recently updated first. */
export function listConversations(): Promise<ConversationSummary[]> {
  return request("/api/conversations");
}

/** Create a conversation (auto-titled by the first chat turn unless named). */
export function createConversation(
  title?: string,
): Promise<ConversationSummary> {
  return request("/api/conversations", {
    method: "POST",
    body: JSON.stringify({ title: title ?? null }),
  });
}

/** Fetch a full transcript (messages plus each answer's stored sources). */
export function getConversation(id: string): Promise<ConversationDetail> {
  return request(`/api/conversations/${encodeURIComponent(id)}`);
}

/** Delete a conversation; its messages and sources cascade server-side. */
export function deleteConversation(id: string): Promise<void> {
  return request(`/api/conversations/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
}

/** Fetch the static capabilities + upload constraints (spec_v2 §4.2). */
export function getConfig(): Promise<ConfigResponse> {
  return request("/api/config");
}

/** Fetch the effective settings catalog + the corpus-stale flag (§4.7). */
export function getSettings(): Promise<SettingsResponse> {
  return request("/api/settings");
}

/**
 * Apply runtime setting overrides (`null` clears one). Query-time knobs
 * take effect on the next question; a reingest-affecting change flips
 * `corpus_stale` in the response.
 */
export function patchSettings(
  overrides: Record<string, SettingValue | null>,
): Promise<SettingsResponse> {
  return request("/api/settings", {
    method: "PATCH",
    body: JSON.stringify({ overrides }),
  });
}

/** List the ingested documents (the corpus-management table). */
export function listDocuments(): Promise<DocumentOut[]> {
  return request("/api/documents");
}

/**
 * Upload files into the corpus directory (no auto-ingest). The browser
 * sets the multipart boundary itself, so no content-type header here.
 * `paths` (folder uploads, spec_v3 §5.2) pairs one relative path per file,
 * positionally — the server rejects the batch when the lengths differ.
 */
export async function uploadDocuments(
  files: File[],
  paths?: readonly string[] | null,
): Promise<UploadResponse> {
  const form = new FormData();
  for (const file of files) form.append("files", file, file.name);
  if (paths != null) for (const path of paths) form.append("paths", path);
  const response = await fetch(`${API_URL}/api/documents`, {
    method: "POST",
    body: form,
  });
  if (!response.ok) throw await toApiError(response);
  return (await response.json()) as UploadResponse;
}

/**
 * Remove documents' chunks from both stores in one round trip (optionally
 * their files too) — the corpus table's delete, for one row or a whole
 * selection. A `POST` sub-path, not a body on `DELETE`.
 *
 * Ids that no longer exist come back in `not_found` rather than failing the
 * batch, so a selection made before a concurrent delete still removes what
 * is left. The single-document `DELETE /api/documents/{doc_id}` route
 * remains the REST shape for API consumers; the GUI has no second delete
 * path of its own.
 */
export function deleteDocuments(
  docIds: readonly string[],
  options?: { removeFile?: boolean },
): Promise<DocumentBulkDeleteResponse> {
  const query = options?.removeFile ? "?remove_file=true" : "";
  return request(`/api/documents/delete${query}`, {
    method: "POST",
    body: JSON.stringify({ doc_ids: docIds }),
  });
}

/**
 * Locate a chunk's text in its source document for the page preview
 * (ADR-010). Every degradable condition (file changed, format
 * unsupported, no text match, …) comes back `200 available:false` with a
 * `reason` — only an unknown `doc_id` rejects.
 */
export function locatePreview(
  docId: string,
  text: string,
): Promise<PreviewLocateResponse> {
  return request(
    `/api/documents/${encodeURIComponent(docId)}/preview/locate`,
    { method: "POST", body: JSON.stringify({ text }) },
  );
}

/**
 * URL of one rendered page image — a plain `<img src>`. The response is
 * `Cache-Control: immutable` (sound: `doc_id` is content-hashed), so the
 * browser is the cache layer and no query machinery is involved.
 */
export function previewPageUrl(docId: string, page: number): string {
  return `${API_URL}/api/documents/${encodeURIComponent(docId)}/preview/page/${page}`;
}

/** Trigger a background ingest run; 409s while one is in flight. */
export function startIngest(reingest: boolean): Promise<IngestRun> {
  return request("/api/ingest", {
    method: "POST",
    body: JSON.stringify({ reingest }),
  });
}

/**
 * One parsed frame of the chat SSE protocol, discriminated on the event
 * name (spec_v2 §4.3): `retrieval` → `reasoning`/`token` deltas → `done`,
 * with `error` as the in-band mid-stream failure. Throttled `stats`
 * frames (live decode throughput) interleave with the deltas — but only
 * when the model server reports its own timings (llama.cpp does), so a
 * turn with zero `stats` frames is normal, not stalled.
 */
export type ChatEvent =
  | { type: "retrieval"; data: RetrievalEvent }
  | { type: "reasoning"; data: DeltaEvent }
  | { type: "token"; data: DeltaEvent }
  | { type: "stats"; data: StatsEvent }
  | { type: "done"; data: DoneEvent }
  | { type: "error"; data: ChatErrorEvent };

const CHAT_EVENT_NAMES = new Set([
  "retrieval",
  "reasoning",
  "token",
  "stats",
  "done",
  "error",
]);

/**
 * One parsed frame of the ingest-status SSE protocol (spec_v2 §4.2):
 * a `status` snapshot first, `progress`/`log` frames while running, and a
 * terminal `status` carrying the summary.
 */
export type IngestEvent =
  | { type: "status"; data: IngestStatusEvent }
  | { type: "progress"; data: IngestProgressEvent }
  | { type: "log"; data: IngestLogEvent };

const INGEST_EVENT_NAMES = new Set(["status", "progress", "log"]);

/**
 * Parse a raw SSE byte stream into `{type, data}` events.
 *
 * `eventsource-parser` owns the framing (partial lines buffered across
 * chunks, multi-line `data:`, comments ignored); this layer keeps only the
 * named events in `names` and JSON-decodes their payloads. Unknown event
 * names are skipped so the protocols can grow without breaking old clients.
 */
async function* parseNamedSSE(
  body: ReadableStream<Uint8Array>,
  names: ReadonlySet<string>,
): AsyncGenerator<{ type: string; data: unknown }, void, undefined> {
  // TextDecoderStream's writable side is typed WritableStream<BufferSource>;
  // TS's invariant stream generics reject the (safe) Uint8Array pipe.
  const decoder = new TextDecoderStream() as unknown as ReadableWritablePair<
    string,
    Uint8Array
  >;
  const frames = body
    .pipeThrough(decoder)
    .pipeThrough(new EventSourceParserStream());
  const reader = frames.getReader();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) return;
      if (!value.event || !names.has(value.event)) continue;
      yield { type: value.event, data: JSON.parse(value.data) };
    }
  } finally {
    reader.releaseLock();
  }
}

/** Parse a raw SSE byte stream into typed chat {@link ChatEvent}s. */
export async function* parseSSE(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<ChatEvent, void, undefined> {
  yield* parseNamedSSE(body, CHAT_EVENT_NAMES) as AsyncGenerator<
    ChatEvent,
    void,
    undefined
  >;
}

/**
 * Follow the current (or last) ingest run over `GET /api/ingest/status`.
 *
 * The stream replays the run's events from the start, then follows live
 * until the terminal `status` frame — connecting mid-run or after
 * completion renders the same picture. With no run it is a single idle
 * `status` frame.
 */
export async function* streamIngestStatus(
  signal?: AbortSignal,
): AsyncGenerator<IngestEvent, void, undefined> {
  const response = await fetch(`${API_URL}/api/ingest/status`, { signal });
  if (!response.ok) throw await toApiError(response);
  if (!response.body) {
    throw new ApiError(response.status, "empty_body", "The status response carried no stream.");
  }
  yield* parseNamedSSE(response.body, INGEST_EVENT_NAMES) as AsyncGenerator<
    IngestEvent,
    void,
    undefined
  >;
}

/**
 * Ask one question over `POST /api/chat` and yield its SSE events.
 *
 * Failures *before* the stream opens (503 dependency down, 404 unknown
 * conversation, 422) throw an {@link ApiError}; failures after arrive as an
 * in-band `error` event. Abort the `signal` to stop generation server-side
 * (the flow notices between tokens and frees the GPU).
 */
export async function* streamChat(
  requestBody: ChatRequest,
  signal?: AbortSignal,
): AsyncGenerator<ChatEvent, void, undefined> {
  const response = await fetch(`${API_URL}/api/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(requestBody),
    signal,
  });
  if (!response.ok) throw await toApiError(response);
  if (!response.body) {
    throw new ApiError(response.status, "empty_body", "The chat response carried no stream.");
  }
  yield* parseSSE(response.body);
}
