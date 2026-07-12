import { describe, expect, it } from "vitest";

import type { RetrievalTrace } from "@/lib/api";
import { buildTraceBadges, formatDelta } from "@/lib/trace";

function makeTrace(overrides: Partial<RetrievalTrace> = {}): RetrievalTrace {
  return {
    semantic_rank: null,
    semantic_score: null,
    bm25_rank: null,
    bm25_score: null,
    fused_score: 0.94,
    fused_rank: 2,
    rerank_score: null,
    rerank_delta: null,
    final_rank: 2,
    ...overrides,
  };
}

function labels(trace: RetrievalTrace): string[] {
  return buildTraceBadges(trace).map((badge) => badge.label);
}

describe("buildTraceBadges", () => {
  it("renders both arms plus fused when both surfaced the chunk", () => {
    const trace = makeTrace({
      semantic_rank: 1,
      semantic_score: 0.8321,
      bm25_rank: 3,
      bm25_score: 12.4,
    });
    expect(labels(trace)).toEqual(["sem #1", "bm25 #3", "fused 0.94"]);
  });

  it("collapses a missing bm25 arm into a semantic-only badge", () => {
    const trace = makeTrace({ semantic_rank: 4, semantic_score: 0.71 });
    const badges = buildTraceBadges(trace);
    expect(badges.map((badge) => badge.kind)).toEqual([
      "semantic-only",
      "fused",
    ]);
    expect(badges[0].label).toBe("semantic-only #4");
    expect(badges[0].tone).toBe("muted");
  });

  it("collapses a missing semantic arm into a bm25-only badge", () => {
    const trace = makeTrace({ bm25_rank: 2, bm25_score: 9.1 });
    expect(labels(trace)).toEqual(["bm25-only #2", "fused 0.94"]);
  });

  it("renders no arm badge when neither arm reported a rank", () => {
    expect(labels(makeTrace())).toEqual(["fused 0.94"]);
  });

  it("formats the fused score to two decimals (CLI parity)", () => {
    expect(labels(makeTrace({ fused_score: 0.0421 }))).toContain("fused 0.04");
    expect(labels(makeTrace({ fused_score: 1 }))).toContain("fused 1.00");
  });

  it("adds a signed rerank badge only when the rerank stage ran", () => {
    expect(labels(makeTrace())).not.toContain("rerank +0");

    const up = buildTraceBadges(
      makeTrace({ rerank_delta: 2, rerank_score: 0.9973 }),
    ).at(-1);
    expect(up?.label).toBe("rerank +2");
    expect(up?.tone).toBe("up");

    const down = buildTraceBadges(makeTrace({ rerank_delta: -1 })).at(-1);
    expect(down?.label).toBe("rerank -1");
    expect(down?.tone).toBe("down");

    const flat = buildTraceBadges(makeTrace({ rerank_delta: 0 })).at(-1);
    expect(flat?.label).toBe("rerank +0");
    expect(flat?.tone).toBe("neutral");
  });

  it("hover details carry the underlying scores", () => {
    const badges = buildTraceBadges(
      makeTrace({
        semantic_rank: 1,
        semantic_score: 0.8321,
        bm25_rank: 3,
        bm25_score: 12.4,
        rerank_delta: 2,
        rerank_score: 0.9973,
      }),
    );
    expect(badges.find((b) => b.kind === "semantic")?.detail).toContain(
      "0.8321",
    );
    expect(badges.find((b) => b.kind === "bm25")?.detail).toContain("12.40");
    expect(badges.find((b) => b.kind === "fused")?.detail).toContain("#2");
    expect(badges.find((b) => b.kind === "rerank")?.detail).toContain(
      "0.9973",
    );
  });
});

describe("formatDelta", () => {
  it("always shows the sign", () => {
    expect(formatDelta(3)).toBe("+3");
    expect(formatDelta(-2)).toBe("-2");
    expect(formatDelta(0)).toBe("+0");
  });
});
