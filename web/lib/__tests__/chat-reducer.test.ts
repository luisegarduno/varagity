import { describe, expect, it } from "vitest";

import type { ChatEvent, DoneEvent, RetrievalEvent } from "@/lib/api";
import { newTurn, reduceChatEvent, type StreamingTurn } from "@/lib/chat-reducer";

const retrieval: RetrievalEvent = {
  chunks: [],
  method: "reranked",
  top_k: 10,
  reranked_to: 5,
  condensed_query: null,
};

const done: DoneEvent = {
  message_id: "m1",
  conversation_id: "c1",
  answer: "The full authoritative answer.",
  usage: {
    prompt_tokens: 10,
    completion_tokens: 5,
    latency_ms: { total: 42 },
    tokens_per_second: 56.1,
  },
};

function play(events: ChatEvent[], start?: StreamingTurn): StreamingTurn {
  return events.reduce(reduceChatEvent, start ?? newTurn("q"));
}

describe("reduceChatEvent", () => {
  it("starts empty, keeping the query for the optimistic user bubble", () => {
    expect(newTurn("why?")).toEqual({
      query: "why?",
      reasoning: "",
      answer: "",
      retrieval: null,
      tokensPerSecond: null,
      done: null,
      error: null,
      stopped: false,
    });
  });

  it("accumulates token deltas in arrival order", () => {
    const turn = play([
      { type: "token", data: { delta: "Kelp" } },
      { type: "token", data: { delta: " corridor" } },
      { type: "token", data: { delta: "." } },
    ]);
    expect(turn.answer).toBe("Kelp corridor.");
  });

  it("keeps reasoning separate from the answer", () => {
    const turn = play([
      { type: "reasoning", data: { delta: "think " } },
      { type: "reasoning", data: { delta: "hard" } },
      { type: "token", data: { delta: "answer" } },
    ]);
    expect(turn.reasoning).toBe("think hard");
    expect(turn.answer).toBe("answer");
  });

  it("stashes the retrieval payload without touching the text", () => {
    const turn = play([{ type: "retrieval", data: retrieval }]);
    expect(turn.retrieval).toEqual(retrieval);
    expect(turn.answer).toBe("");
  });

  it("on done, adopts the authoritative full answer over streamed deltas", () => {
    const turn = play([
      { type: "token", data: { delta: "partial gl" } },
      { type: "done", data: done },
    ]);
    expect(turn.done).toEqual(done);
    expect(turn.answer).toBe("The full authoritative answer.");
  });

  it("tracks the newest stats frame's rate (cumulative — latest wins)", () => {
    const turn = play([
      { type: "stats", data: { tokens_per_second: 61.2, completion_tokens: 10 } },
      { type: "stats", data: { tokens_per_second: 55.8, completion_tokens: 40 } },
    ]);
    expect(turn.tokensPerSecond).toBe(55.8);
  });

  it("on done, the final rate supersedes the last throttled stats frame", () => {
    const turn = play([
      { type: "stats", data: { tokens_per_second: 61.2, completion_tokens: 10 } },
      { type: "done", data: done },
    ]);
    expect(turn.tokensPerSecond).toBe(56.1);
  });

  it("on done without a rate, clears any stale live value", () => {
    const noRate = { ...done, usage: { ...done.usage, tokens_per_second: null } };
    const turn = play([
      { type: "stats", data: { tokens_per_second: 61.2, completion_tokens: 10 } },
      { type: "done", data: noRate },
    ]);
    expect(turn.tokensPerSecond).toBeNull();
  });

  it("captures the in-band error event", () => {
    const turn = play([
      { type: "token", data: { delta: "so far" } },
      { type: "error", data: { code: "pipeline_error", message: "boom" } },
    ]);
    expect(turn.error).toEqual({ code: "pipeline_error", message: "boom" });
    expect(turn.answer).toBe("so far");
  });

  it("does not mutate the previous state", () => {
    const first = newTurn("q");
    const second = reduceChatEvent(first, {
      type: "token",
      data: { delta: "x" },
    });
    expect(first.answer).toBe("");
    expect(second.answer).toBe("x");
  });
});
