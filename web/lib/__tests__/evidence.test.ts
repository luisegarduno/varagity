import { describe, expect, it } from "vitest";

import type {
  ChatMessage,
  DoneEvent,
  RetrievalEvent,
  RetrievedChunk,
} from "@/lib/api";
import {
  assistantMessageFromTurn,
  docIdFromChunkId,
  evidenceFromMessage,
  evidenceFromRetrieval,
  fileClock,
  formatTokensPerSecond,
  latencyRecord,
  LIVE_EVIDENCE_KEY,
  usageFromDone,
} from "@/lib/evidence";

// Realistic 16-hex doc ids: the persisted path re-derives `docId` from the
// `{doc_id}::{index}` chunk id, so the round-trip test needs parseable ones.
const chunkA: RetrievedChunk = {
  chunk_id: "a398491c7441925f::0",
  doc_id: "a398491c7441925f",
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
    file_created_at: "2024-01-02T08:00:00Z",
    file_modified_at: "2024-05-04T12:30:45Z",
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
  chunk_id: "174928b2d662c122::4",
  doc_id: "174928b2d662c122",
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
  condensed_query: null,
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
      key: "a398491c7441925f::0",
      docId: "a398491c7441925f",
      rank: 1,
      score: 0.9973,
      fileName: "kelp_corridor.md",
      fileType: "md",
      page: null,
      extraction: "text",
      fileCreatedAt: "2024-01-02T08:00:00Z",
      fileModifiedAt: "2024-05-04T12:30:45Z",
    });
    expect(first.trace?.rerank_delta).toBe(1);
    // chunkB predates the timestamp fields — they normalize to null.
    expect(second).toMatchObject({
      key: "174928b2d662c122::4",
      docId: "174928b2d662c122",
      rank: 2,
      page: 7,
      extraction: "ocr_fallback",
      fileCreatedAt: null,
      fileModifiedAt: null,
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
      chunk_id: "a398491c7441925f::0",
    });
    expect(message.sources[0].trace).toMatchObject({
      score: 0.9973,
      source: "/docs/marine/kelp_corridor.md",
      file_name: "kelp_corridor.md",
      extraction: "text",
      file_created_at: "2024-01-02T08:00:00Z",
      file_modified_at: "2024-05-04T12:30:45Z",
    });

    // The persisted view renders the same chunks as the live view did —
    // including `docId`, which the snapshot path re-derives from `chunk_id`
    // (snapshots never stored `doc_id`; history previews ride on this).
    const persisted = evidenceFromMessage(message, "kelp corridor");
    const live = evidenceFromRetrieval(retrieval, { query: "kelp corridor" });
    expect(persisted?.chunks).toEqual(live.chunks);
    expect(persisted?.chunks[0].docId).toBe("a398491c7441925f");
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
      docId: null,
      rank: 1,
      score: null,
      content: "",
      page: null,
      trace: null,
    });
  });
});

describe("condensedQuery threading (spec_v3 §4.7)", () => {
  const condensed: RetrievalEvent = {
    ...retrieval,
    condensed_query: "kelp corridor length between Bruma and Cinza",
  };

  it("maps the live event's rewrite, and null when searched verbatim", () => {
    expect(evidenceFromRetrieval(condensed).condensedQuery).toBe(
      "kelp corridor length between Bruma and Cinza",
    );
    expect(evidenceFromRetrieval(retrieval).condensedQuery).toBeNull();
  });

  it("folds the rewrite into the turn's message so reloads render the same", () => {
    const message = assistantMessageFromTurn(done, condensed, "");
    expect(message.condensed_query).toBe(
      "kelp corridor length between Bruma and Cinza",
    );
    expect(evidenceFromMessage(message, null)?.condensedQuery).toBe(
      "kelp corridor length between Bruma and Cinza",
    );
    // Verbatim searches stay null through the same fold.
    expect(
      evidenceFromMessage(assistantMessageFromTurn(done, retrieval, ""), null)
        ?.condensedQuery,
    ).toBeNull();
  });

  it("treats persisted pre-v3 messages (no field at all) as verbatim", () => {
    const legacy: ChatMessage = {
      message_id: "a3",
      role: "assistant",
      content: "answer",
      created_at: "2026-07-12T00:00:00Z",
      sources: [
        { rank: 1, chunk_id: "a398491c7441925f::0", trace: { content: "c" } },
      ],
    };
    expect(evidenceFromMessage(legacy, null)?.condensedQuery).toBeNull();
  });
});

describe("fileClock", () => {
  const modifiedIso = "2024-05-04T12:30:45Z";
  const createdIso = "2024-01-02T08:00:00Z";

  it("renders the modified date, with both full stamps in the tooltip", () => {
    const clock = fileClock({
      fileCreatedAt: createdIso,
      fileModifiedAt: modifiedIso,
    });
    // Locale-dependent output: build the expectation with the same API.
    expect(clock?.text).toBe(
      `modified ${new Date(modifiedIso).toLocaleDateString()}`,
    );
    expect(clock?.title).toBe(
      `Source file modified ${new Date(modifiedIso).toLocaleString()}` +
        ` · created ${new Date(createdIso).toLocaleString()}`,
    );
  });

  it("omits the birth time when the filesystem recorded none", () => {
    const clock = fileClock({ fileCreatedAt: null, fileModifiedAt: modifiedIso });
    expect(clock?.title).toBe(
      `Source file modified ${new Date(modifiedIso).toLocaleString()}`,
    );
  });

  it("returns null for pre-timestamp chunks and unparseable values", () => {
    expect(fileClock({ fileCreatedAt: null, fileModifiedAt: null })).toBeNull();
    // A birth time alone is not worth a segment the line sorts by mtime.
    expect(
      fileClock({ fileCreatedAt: createdIso, fileModifiedAt: null }),
    ).toBeNull();
    expect(
      fileClock({ fileCreatedAt: null, fileModifiedAt: "not-a-date" }),
    ).toBeNull();
  });
});

describe("docIdFromChunkId", () => {
  it("extracts the 16-hex doc id prefix", () => {
    expect(docIdFromChunkId("a398491c7441925f::7")).toBe("a398491c7441925f");
  });

  it("rejects prefixes that are not a content-hash doc id", () => {
    expect(docIdFromChunkId("doc9::1")).toBeNull(); // not hex-16
    expect(docIdFromChunkId("A398491C7441925F::7")).toBeNull(); // wrong case
    expect(docIdFromChunkId("a398491c7441925f00::1")).toBeNull(); // too long
    expect(docIdFromChunkId("")).toBeNull();
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
