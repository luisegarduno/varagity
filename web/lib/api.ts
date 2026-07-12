/**
 * Typed client for the Varagity HTTP API (spec_v2 §4).
 *
 * Every wire shape is re-exported from the generated `lib/types.ts`
 * (`pnpm gen:types` against the live `/openapi.json`) — nothing here is
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
export type DoneEvent = Schemas["DoneEvent"];
export type ChatErrorEvent = Schemas["ErrorEvent"];
export type ErrorResponse = Schemas["ErrorResponse"];
export type HealthResponse = Schemas["HealthResponse"];

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

/**
 * One parsed frame of the chat SSE protocol, discriminated on the event
 * name (spec_v2 §4.3): `retrieval` → `reasoning`/`token` deltas → `done`,
 * with `error` as the in-band mid-stream failure.
 */
export type ChatEvent =
  | { type: "retrieval"; data: RetrievalEvent }
  | { type: "reasoning"; data: DeltaEvent }
  | { type: "token"; data: DeltaEvent }
  | { type: "done"; data: DoneEvent }
  | { type: "error"; data: ChatErrorEvent };

const CHAT_EVENT_NAMES = new Set([
  "retrieval",
  "reasoning",
  "token",
  "done",
  "error",
]);

/**
 * Parse a raw SSE byte stream into typed {@link ChatEvent}s.
 *
 * `eventsource-parser` owns the framing (partial lines buffered across
 * chunks, multi-line `data:`, comments ignored); this layer keeps only the
 * protocol's named events and JSON-decodes their payloads. Unknown event
 * names are skipped so the protocol can grow without breaking old clients.
 */
export async function* parseSSE(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<ChatEvent, void, undefined> {
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
      if (!value.event || !CHAT_EVENT_NAMES.has(value.event)) continue;
      yield {
        type: value.event,
        data: JSON.parse(value.data),
      } as ChatEvent;
    }
  } finally {
    reader.releaseLock();
  }
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
