import { beforeEach, describe, expect, it } from "vitest";

import {
  clearSessionUsage,
  recordSessionUsage,
  sessionUsage,
} from "@/lib/session-usage";

describe("session usage recall", () => {
  beforeEach(() => {
    clearSessionUsage();
  });

  it("returns what was recorded for a message id, normalized", () => {
    recordSessionUsage("m1", {
      prompt_tokens: 104,
      completion_tokens: 24,
      latency_ms: { total: 1420 },
      tokens_per_second: 56.07,
    });
    expect(sessionUsage("m1")).toEqual({
      promptTokens: 104,
      completionTokens: 24,
      tokensPerSecond: 56.07,
    });
  });

  it("returns null for turns never recorded (pre-reload history)", () => {
    expect(sessionUsage("m-from-last-week")).toBeNull();
  });

  it("skips recording when the server reported nothing usable", () => {
    recordSessionUsage("m2", {
      prompt_tokens: null,
      completion_tokens: null,
      latency_ms: { total: 10 },
      tokens_per_second: null,
    });
    expect(sessionUsage("m2")).toBeNull();
  });

  it("keeps counts even when the server reports no rate (non-llama.cpp)", () => {
    recordSessionUsage("m3", {
      prompt_tokens: 9,
      completion_tokens: 3,
      latency_ms: {},
      tokens_per_second: null,
    });
    expect(sessionUsage("m3")).toEqual({
      promptTokens: 9,
      completionTokens: 3,
      tokensPerSecond: null,
    });
  });
});
