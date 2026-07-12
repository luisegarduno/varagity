import { describe, expect, it } from "vitest";

import { parseSSE, type ChatEvent } from "@/lib/api";

/** A byte stream delivering each piece as its own chunk. */
function byteStream(...pieces: (string | Uint8Array)[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const piece of pieces) {
        controller.enqueue(
          typeof piece === "string" ? encoder.encode(piece) : piece,
        );
      }
      controller.close();
    },
  });
}

async function collect(stream: ReadableStream<Uint8Array>): Promise<ChatEvent[]> {
  const events: ChatEvent[] = [];
  for await (const event of parseSSE(stream)) events.push(event);
  return events;
}

describe("parseSSE", () => {
  it("dispatches the protocol's named events in order with parsed payloads", async () => {
    const events = await collect(
      byteStream(
        'event: retrieval\ndata: {"chunks": [], "method": "reranked", "top_k": 10, "reranked_to": 5}\n\n',
        'event: reasoning\ndata: {"delta": "hmm"}\n\n',
        'event: token\ndata: {"delta": "Kelp"}\n\n',
        'event: token\ndata: {"delta": " corridor"}\n\n',
        'event: done\ndata: {"message_id": "m1", "conversation_id": "c1", "answer": "Kelp corridor", "usage": {"prompt_tokens": 3, "completion_tokens": 2, "latency_ms": {"total": 5}}}\n\n',
      ),
    );

    expect(events.map((event) => event.type)).toEqual([
      "retrieval",
      "reasoning",
      "token",
      "token",
      "done",
    ]);
    expect(events[0].data).toMatchObject({ method: "reranked", reranked_to: 5 });
    expect(events[2].data).toEqual({ delta: "Kelp" });
    if (events[4].type !== "done") throw new Error("expected done");
    expect(events[4].data.answer).toBe("Kelp corridor");
  });

  it("buffers frames split across chunk boundaries (mid-line, mid-frame)", async () => {
    const events = await collect(
      byteStream(
        "event: retriev",
        'al\ndata: {"chunks": [], "met',
        'hod": "hybrid", "top_k": 3, "reranked_to": null}\n',
        '\nevent: token\ndata: {"del',
        'ta": "Hi"}\n\n',
      ),
    );

    expect(events.map((event) => event.type)).toEqual(["retrieval", "token"]);
    expect(events[0].data).toMatchObject({ method: "hybrid", top_k: 3 });
    expect(events[1].data).toEqual({ delta: "Hi" });
  });

  it("reassembles multi-byte UTF-8 split across chunks", async () => {
    const frame = new TextEncoder().encode(
      'event: token\ndata: {"delta": "café"}\n\n',
    );
    const splitAt = frame.indexOf(0xc3) + 1; // inside the 2-byte "é"
    const events = await collect(
      byteStream(frame.slice(0, splitAt), frame.slice(splitAt)),
    );

    expect(events).toEqual([{ type: "token", data: { delta: "café" } }]);
  });

  it("skips unknown event names, unnamed frames, and comments", async () => {
    const events = await collect(
      byteStream(
        ": heartbeat comment\n\n",
        'data: {"orphan": true}\n\n',
        'event: shiny_new_event\ndata: {"x": 1}\n\n',
        'event: token\ndata: {"delta": "ok"}\n\n',
      ),
    );

    expect(events).toEqual([{ type: "token", data: { delta: "ok" } }]);
  });

  it("surfaces the in-band error event", async () => {
    const events = await collect(
      byteStream(
        'event: error\ndata: {"code": "pipeline_error", "message": "boom"}\n\n',
      ),
    );

    expect(events).toEqual([
      { type: "error", data: { code: "pipeline_error", message: "boom" } },
    ]);
  });
});
