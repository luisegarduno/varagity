import { describe, expect, it } from "vitest";

import type { ChatEvent, DoneEvent, RetrievalEvent } from "@/lib/api";
import { newTurn, reduceChatEvent, type StreamingTurn } from "@/lib/chat-reducer";

const retrieval: RetrievalEvent = {
  chunks: [],
  method: "reranked",
  top_k: 10,
  reranked_to: 5,
};

const done: DoneEvent = {
  message_id: "m1",
  conversation_id: "c1",
  answer: "The full authoritative answer.",
  usage: { prompt_tokens: 10, completion_tokens: 5, latency_ms: { total: 42 } },
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
