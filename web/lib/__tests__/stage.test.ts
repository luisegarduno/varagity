import { describe, expect, it } from "vitest";

import { currentStage, deriveStages, type StageTurn } from "@/lib/stage";

function makeTurn(overrides: Partial<StageTurn> = {}): StageTurn {
  return {
    reasoning: "",
    answer: "",
    retrieval: null,
    done: null,
    error: null,
    stopped: false,
    ...overrides,
  };
}

function statuses(turn: StageTurn, rerankActive = false) {
  return deriveStages(turn, { rerankActive }).map((stage) => [
    stage.key,
    stage.status,
  ]);
}

describe("deriveStages", () => {
  it("marks retrieve active while waiting, without a rerank stage", () => {
    expect(statuses(makeTurn())).toEqual([
      ["retrieve", "active"],
      ["generate", "pending"],
    ]);
  });

  it("shows a pending rerank stage while waiting when the settings say so", () => {
    expect(statuses(makeTurn(), true)).toEqual([
      ["retrieve", "active"],
      ["rerank", "pending"],
      ["generate", "pending"],
    ]);
  });

  it("completes retrieve and rerank together when the retrieval event lands", () => {
    // The server emits `retrieval` post-rerank — one event, both stages.
    const turn = makeTurn({ retrieval: { top_k: 40, reranked_to: 5 } });
    expect(statuses(turn)).toEqual([
      ["retrieve", "done"],
      ["rerank", "done"],
      ["generate", "active"],
    ]);
  });

  it("trusts the retrieval event over the settings guess", () => {
    // Settings said reranked, but the event reports no narrowing: no stage.
    const turn = makeTurn({ retrieval: { top_k: 10, reranked_to: null } });
    expect(statuses(turn, true)).toEqual([
      ["retrieve", "done"],
      ["generate", "active"],
    ]);
  });

  it("carries the narrowing detail on the rerank stage", () => {
    const turn = makeTurn({ retrieval: { top_k: 40, reranked_to: 5 } });
    const rerank = deriveStages(turn, { rerankActive: false })[1];
    expect(rerank.key).toBe("rerank");
    expect(rerank.detail).toBe("40 → 5");
  });

  it("leaves the rerank detail empty until the event lands", () => {
    const rerank = deriveStages(makeTurn(), { rerankActive: true })[1];
    expect(rerank.detail).toBeNull();
  });

  it("treats a delta before the retrieval event as retrieval done", () => {
    expect(statuses(makeTurn({ reasoning: "hmm" }))).toEqual([
      ["retrieve", "done"],
      ["generate", "active"],
    ]);
    expect(statuses(makeTurn({ answer: "Kelp" }))).toEqual([
      ["retrieve", "done"],
      ["generate", "active"],
    ]);
  });

  it("marks every stage done once the turn completes", () => {
    const turn = makeTurn({
      retrieval: { top_k: 40, reranked_to: 5 },
      answer: "Kelp.",
      done: { message_id: "m1" },
    });
    expect(statuses(turn)).toEqual([
      ["retrieve", "done"],
      ["rerank", "done"],
      ["generate", "done"],
    ]);
  });

  it("leaves nothing active on a stopped turn", () => {
    const turn = makeTurn({
      retrieval: { top_k: 10, reranked_to: null },
      answer: "partial",
      stopped: true,
    });
    expect(statuses(turn)).toEqual([
      ["retrieve", "done"],
      ["generate", "pending"],
    ]);
    expect(currentStage(deriveStages(turn, { rerankActive: false }))).toBeNull();
  });

  it("fails the retrieve stage on an error before the retrieval event", () => {
    const turn = makeTurn({ error: { code: "es_unreachable", message: "down" } });
    expect(statuses(turn, true)).toEqual([
      ["retrieve", "failed"],
      ["rerank", "pending"],
      ["generate", "pending"],
    ]);
  });

  it("fails the generate stage on an error after the retrieval event", () => {
    const turn = makeTurn({
      retrieval: { top_k: 40, reranked_to: 5 },
      error: { code: "pipeline_error", message: "boom" },
    });
    expect(statuses(turn)).toEqual([
      ["retrieve", "done"],
      ["rerank", "done"],
      ["generate", "failed"],
    ]);
  });
});

describe("currentStage", () => {
  it("returns the active stage while streaming", () => {
    const stages = deriveStages(makeTurn(), { rerankActive: false });
    expect(currentStage(stages)?.key).toBe("retrieve");
  });

  it("returns the failed stage on an error", () => {
    const turn = makeTurn({
      retrieval: { top_k: 40, reranked_to: 5 },
      error: { code: "pipeline_error", message: "boom" },
    });
    const stages = deriveStages(turn, { rerankActive: false });
    expect(currentStage(stages)?.key).toBe("generate");
    expect(currentStage(stages)?.status).toBe("failed");
  });
});
