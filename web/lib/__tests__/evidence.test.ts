import { describe, expect, it } from "vitest";

import type {
  ChatMessage,
  DoneEvent,
  RetrievalEvent,
  RetrievedChunk,
} from "@/lib/api";
import {
  assistantMessageFromTurn,
  evidenceFromMessage,
  evidenceFromRetrieval,
  formatTokensPerSecond,
  latencyRecord,
  LIVE_EVIDENCE_KEY,
  usageFromDone,
} from "@/lib/evidence";

const chunkA: RetrievedChunk = {
  chunk_id: "doc1::0",
  doc_id: "doc1",
  original_index: 0,
  content: "Kelp corridors shelter juvenile fish.",
  context: "From a survey of the Aurora coast.",
  metadata: {
    source: "/docs/marine/kelp_corridor.md",
    file_name: "kelp_corridor.md",
    file_type: "md",
    page: null,
    extraction: "text",
    chunking_strategy: "simple",
  },
  score: 0.9973,
  trace: {
    semantic_rank: 1,
    semantic_score: 0.83,
    bm25_rank: 3,
    bm25_score: 12.4,
    fused_score: 0.94,
    fused_rank: 2,
    rerank_score: 0.9973,
    rerank_delta: 1,
    final_rank: 1,
  },
};

const chunkB: RetrievedChunk = {
  chunk_id: "doc2::4",
  doc_id: "doc2",
  original_index: 9,
  content: "Scanned page content.",
  context: null,
  metadata: {
    source: "/docs/scans/survey.pdf",
    file_name: "survey.pdf",
    file_type: "pdf",
    page: 7,
    extraction: "ocr_fallback",
  },
  score: 0.41,
  trace: null,
};

const retrieval: RetrievalEvent = {
  chunks: [chunkA, chunkB],
  method: "reranked",
  top_k: 10,
  reranked_to: 5,
};

const done: DoneEvent = {
  message_id: "m42",
  conversation_id: "c7",
  answer: "Kelp corridors shelter juvenile fish [SOURCE]: /docs/marine/kelp_corridor.md",
  usage: {
    prompt_tokens: 100,
    completion_tokens: 20,
    latency_ms: { retrieval: 690, generation: 8441, total: 9200 },
    tokens_per_second: 2.4,
  },
};

describe("evidenceFromRetrieval", () => {
  it("normalizes the live event: ranks by position, provenance from metadata", () => {
    const evidence = evidenceFromRetrieval(retrieval, {
      query: "kelp corridor",
      latencyMs: done.usage.latency_ms,
    });
    expect(evidence.key).toBe(LIVE_EVIDENCE_KEY);
    expect(evidence.method).toBe("reranked");
    expect(evidence.topK).toBe(10);
    expect(evidence.rerankedTo).toBe(5);
    expect(evidence.latencyMs).toEqual({
      retrieval: 690,
      generation: 8441,
      total: 9200,
    });
    expect(evidence.chunks).toHaveLength(2);

    const [first, second] = evidence.chunks;
    expect(first).toMatchObject({
      key: "doc1::0",
      rank: 1,
      score: 0.9973,
      fileName: "kelp_corridor.md",
      fileType: "md",
      page: null,
      extraction: "text",
    });
    expect(first.trace?.rerank_delta).toBe(1);
    expect(second).toMatchObject({
      key: "doc2::4",
      rank: 2,
      page: 7,
      extraction: "ocr_fallback",
      trace: null,
    });
  });
});

describe("sourcesFromRetrieval / evidenceFromMessage", () => {
  it("mirrors the server snapshot and round-trips through a message", () => {
    const message = assistantMessageFromTurn(done, retrieval, "thinking…");
    expect(message).toMatchObject({
      message_id: "m42",
      role: "assistant",
      content: done.answer,
      retrieval_method: "reranked",
      reasoning: "thinking…",
    });
    expect(message.sources).toHaveLength(2);
    expect(message.sources[0]).toMatchObject({
      rank: 1,
      chunk_id: "doc1::0",
    });
    expect(message.sources[0].trace).toMatchObject({
      score: 0.9973,
      source: "/docs/marine/kelp_corridor.md",
      file_name: "kelp_corridor.md",
      extraction: "text",
    });

    // The persisted view renders the same chunks as the live view did.
    const persisted = evidenceFromMessage(message, "kelp corridor");
    const live = evidenceFromRetrieval(retrieval, { query: "kelp corridor" });
    expect(persisted?.chunks).toEqual(live.chunks);
    expect(persisted?.method).toBe("reranked");
    expect(persisted?.latencyMs).toEqual(done.usage.latency_ms);
    expect(persisted?.query).toBe("kelp corridor");
  });

  it("captures no reasoning field for an empty reasoning stream", () => {
    const message = assistantMessageFromTurn(done, retrieval, "");
    expect(message.reasoning).toBeNull();
  });

  it("returns null for user turns and evidence-less assistant turns", () => {
    const user: ChatMessage = {
      message_id: "u1",
      role: "user",
      content: "why?",
      created_at: "2026-07-12T00:00:00Z",
      sources: [],
    };
    expect(evidenceFromMessage(user, null)).toBeNull();

    const bare: ChatMessage = { ...user, message_id: "a1", role: "assistant" };
    expect(evidenceFromMessage(bare, null)).toBeNull();
  });

  it("tolerates malformed snapshot fields from the JSONB blob", () => {
    const message: ChatMessage = {
      message_id: "a2",
      role: "assistant",
      content: "answer",
      created_at: "2026-07-12T00:00:00Z",
      sources: [
        {
          rank: 1,
          chunk_id: "doc9::1",
          trace: {
            score: "not-a-number",
            content: 42,
            context: null,
            source: null,
            file_name: null,
            file_type: null,
            page: "7",
            extraction: null,
            trace: { fused_score: "bad" },
          },
        },
      ],
    };
    const evidence = evidenceFromMessage(message, null);
    expect(evidence?.chunks[0]).toMatchObject({
      key: "doc9::1",
      rank: 1,
      score: null,
      content: "",
      page: null,
      trace: null,
    });
  });
});

describe("latencyRecord", () => {
  it("keeps numeric stages and drops junk", () => {
    expect(latencyRecord({ retrieval: 690, generation: "x" })).toEqual({
      retrieval: 690,
    });
    expect(latencyRecord(null)).toBeNull();
    expect(latencyRecord({})).toBeNull();
  });
});

describe("usageFromDone", () => {
  it("normalizes the done event's usage block", () => {
    expect(usageFromDone(done.usage)).toEqual({
      promptTokens: 100,
      completionTokens: 20,
      tokensPerSecond: 2.4,
    });
  });

  it("collapses an all-null report to null (nothing worth a footer line)", () => {
    expect(
      usageFromDone({
        prompt_tokens: null,
        completion_tokens: null,
        latency_ms: {},
        tokens_per_second: null,
      }),
    ).toBeNull();
  });

  it("threads into both evidence builders", () => {
    const usage = usageFromDone(done.usage);
    const live = evidenceFromRetrieval(retrieval, { usage });
    expect(live.usage).toEqual(usage);

    const message = assistantMessageFromTurn(done, retrieval, "");
    expect(evidenceFromMessage(message, null, usage)?.usage).toEqual(usage);
    // Without the session-recall arg (a pre-reload turn): no usage.
    expect(evidenceFromMessage(message, null)?.usage).toBeNull();
  });
});

describe("formatTokensPerSecond", () => {
  it("rounds to whole tokens at real decode speeds", () => {
    expect(formatTokensPerSecond(56.07)).toBe("56 tok/s");
    expect(formatTokensPerSecond(199.5)).toBe("200 tok/s");
  });

  it("keeps one decimal below 10 where rounding would hide the number", () => {
    expect(formatTokensPerSecond(2.44)).toBe("2.4 tok/s");
    expect(formatTokensPerSecond(9.96)).toBe("10.0 tok/s");
  });
});
