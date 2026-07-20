/**
 * The token-accumulation reducer: folds the chat SSE events into one
 * in-flight assistant turn.
 *
 * Pure and synchronous so it is trivially unit-testable; the `useChat`
 * hook owns the async loop and calls this per event. The `retrieval`
 * payload is stashed for the evidence panel to consume.
 */
import type {
  ChatErrorEvent,
  ChatEvent,
  DoneEvent,
  RetrievalEvent,
} from "@/lib/api";

/** One streaming assistant turn, accumulated event by event. */
export interface StreamingTurn {
  /** The question this turn answers (rendered as the optimistic user bubble). */
  query: string;
  /** Concatenated `reasoning` deltas (the model's `<think>` stream). */
  reasoning: string;
  /** Concatenated `token` deltas; replaced by `done`'s authoritative answer. */
  answer: string;
  /** The `retrieval` event payload, stashed for the evidence panel. */
  retrieval: RetrievalEvent | null;
  /**
   * Live decode throughput from the newest `stats` frame — the readings
   * are cumulative averages, so the latest supersedes the rest. Stays
   * `null` when the model server reports no timings (anything but
   * llama.cpp), which is what keeps the readout llama.cpp-only.
   */
  tokensPerSecond: number | null;
  /** The terminal `done` payload (ids, full answer, usage), once received. */
  done: DoneEvent | null;
  /** An in-band `error` payload, if the pipeline failed mid-stream. */
  error: ChatErrorEvent | null;
  /** True when the user stopped the stream (the server persists nothing). */
  stopped: boolean;
}

/** A fresh turn for `query`, before any event arrived. */
export function newTurn(query: string): StreamingTurn {
  return {
    query,
    reasoning: "",
    answer: "",
    retrieval: null,
    tokensPerSecond: null,
    done: null,
    error: null,
    stopped: false,
  };
}

/**
 * Fold one SSE event into the turn.
 *
 * `token`/`reasoning` deltas append in arrival order; `done` also replaces
 * the accumulated answer with its authoritative full text (the streamed
 * deltas are best-effort display).
 */
export function reduceChatEvent(
  turn: StreamingTurn,
  event: ChatEvent,
): StreamingTurn {
  switch (event.type) {
    case "retrieval":
      return { ...turn, retrieval: event.data };
    case "reasoning":
      return { ...turn, reasoning: turn.reasoning + event.data.delta };
    case "token":
      return { ...turn, answer: turn.answer + event.data.delta };
    case "stats":
      return { ...turn, tokensPerSecond: event.data.tokens_per_second };
    case "done":
      // `usage.tokens_per_second` is the server's final reading — newer
      // than any throttled `stats` frame, so it wins (and it is `null` on
      // servers that report no timings, clearing a stale live value).
      return {
        ...turn,
        done: event.data,
        answer: event.data.answer,
        tokensPerSecond: event.data.usage.tokens_per_second ?? null,
      };
    case "error":
      return { ...turn, error: event.data };
  }
}
