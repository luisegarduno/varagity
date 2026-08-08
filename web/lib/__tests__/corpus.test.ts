import { describe, expect, it } from "vitest";

import type { ChatMessage } from "@/lib/api";
import {
  asTargetCorpus,
  DEFAULT_CORPUS,
  lastTargetCorpus,
  resolveCorpus,
} from "@/lib/corpus";

function message(
  role: "user" | "assistant",
  overrides: Partial<ChatMessage> = {},
): ChatMessage {
  return {
    message_id: `m${Math.random()}`,
    role,
    content: "…",
    created_at: "2026-08-07T10:00:00Z",
    sources: [],
    ...overrides,
  };
}

describe("asTargetCorpus", () => {
  it("narrows the two known corpora and rejects everything else", () => {
    expect(asTargetCorpus("rag")).toBe("rag");
    expect(asTargetCorpus("graph")).toBe("graph");
    expect(asTargetCorpus(null)).toBeNull();
    expect(asTargetCorpus(undefined)).toBeNull();
    expect(asTargetCorpus("federated")).toBeNull();
  });
});

describe("lastTargetCorpus", () => {
  it("reads the newest assistant turn's requested corpus", () => {
    const transcript = [
      message("user"),
      message("assistant", { target_corpus: "rag" }),
      message("user"),
      message("assistant", { target_corpus: "graph" }),
    ];
    expect(lastTargetCorpus(transcript)).toBe("graph");
  });

  it("reads the request, not the outcome — a degrade keeps the selector", () => {
    // A graph turn that degraded to a chunk answer persists
    // target_corpus="graph" with graph_evidence NULL (ADR-017): the user
    // asked the graph, and the selector must not silently flip back.
    const transcript = [
      message("user"),
      message("assistant", { target_corpus: "graph", graph_evidence: null }),
    ];
    expect(lastTargetCorpus(transcript)).toBe("graph");
  });

  it("ignores user turns, which never carry one", () => {
    const transcript = [
      message("assistant", { target_corpus: "graph" }),
      message("user", { target_corpus: "rag" }),
    ];
    expect(lastTargetCorpus(transcript)).toBe("graph");
  });

  it("skips assistant turns with no (or an unknown) corpus", () => {
    const transcript = [
      message("assistant", { target_corpus: "graph" }),
      message("assistant", { target_corpus: null }), // pre-stage-2 history
      message("assistant", { target_corpus: "federated" }),
    ];
    expect(lastTargetCorpus(transcript)).toBe("graph");
  });

  it("returns null for an empty, all-user, or absent transcript", () => {
    expect(lastTargetCorpus([])).toBeNull();
    expect(lastTargetCorpus([message("user")])).toBeNull();
    expect(lastTargetCorpus(null)).toBeNull();
    expect(lastTargetCorpus(undefined)).toBeNull();
  });
});

describe("resolveCorpus", () => {
  it("prefers an explicit pick over the conversation's history", () => {
    const transcript = [message("assistant", { target_corpus: "graph" })];
    expect(resolveCorpus("rag", transcript)).toBe("rag");
  });

  it("derives from the transcript when nothing was picked here", () => {
    const transcript = [message("assistant", { target_corpus: "graph" })];
    expect(resolveCorpus(null, transcript)).toBe("graph");
  });

  it("falls back to documents for a fresh (or still-loading) conversation", () => {
    expect(resolveCorpus(null, null)).toBe(DEFAULT_CORPUS);
    expect(resolveCorpus(null, [])).toBe("rag");
  });
});
