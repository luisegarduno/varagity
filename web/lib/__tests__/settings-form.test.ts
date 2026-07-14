import { describe, expect, it } from "vitest";

import type { SettingsResponse } from "@/lib/api";
import {
  dirtyFields,
  groupFields,
  initForm,
  patchBody,
  setValue,
  willFlagStale,
} from "@/lib/settings-form";

function catalog(overrides?: Partial<SettingsResponse>): SettingsResponse {
  return {
    settings: [
      {
        name: "RETRIEVAL_METHOD",
        value: "hybrid",
        group: "retrieval",
        overridden: false,
        reingest_affecting: false,
        choices: ["bm25", "hybrid", "reranked", "semantic"],
      },
      {
        name: "TOP_K",
        value: 10,
        group: "retrieval",
        overridden: false,
        reingest_affecting: false,
        choices: null,
      },
      {
        name: "SEMANTIC_WEIGHT",
        value: 0.8,
        group: "retrieval",
        overridden: false,
        reingest_affecting: false,
        choices: null,
      },
      {
        name: "BM25_WEIGHT",
        value: 0.2,
        group: "retrieval",
        overridden: false,
        reingest_affecting: false,
        choices: null,
      },
      {
        name: "LLM_TEMPERATURE",
        value: 0.6,
        group: "generation",
        overridden: true,
        reingest_affecting: false,
        choices: null,
      },
      {
        name: "CHUNKING_STRATEGY",
        value: "recursive_character",
        group: "ingestion",
        overridden: false,
        reingest_affecting: true,
        choices: ["recursive_character", "token_based"],
      },
      {
        name: "CONTEXTUALIZE",
        value: true,
        group: "ingestion",
        overridden: false,
        reingest_affecting: true,
        choices: null,
      },
    ],
    corpus_stale: false,
    ...overrides,
  };
}

describe("initForm", () => {
  it("mirrors the catalog with staged == effective values", () => {
    const form = initForm(catalog());
    expect(form.fields).toHaveLength(7);
    expect(form.corpusStale).toBe(false);
    const topK = form.fields.find((f) => f.name === "TOP_K");
    expect(topK).toMatchObject({ initial: 10, value: 10, overridden: false });
    expect(dirtyFields(form)).toEqual([]);
  });
});

describe("setValue + dirtiness", () => {
  it("stages an edit without touching the initial value", () => {
    const form = setValue(initForm(catalog()), "TOP_K", 25);
    const topK = form.fields.find((f) => f.name === "TOP_K");
    expect(topK).toMatchObject({ initial: 10, value: 25 });
    expect(dirtyFields(form).map((f) => f.name)).toEqual(["TOP_K"]);
  });

  it("re-staging the original value is clean again", () => {
    const form = setValue(setValue(initForm(catalog()), "TOP_K", 25), "TOP_K", 10);
    expect(dirtyFields(form)).toEqual([]);
  });

  it("editing one fusion weight sets the partner to the complement", () => {
    const form = setValue(initForm(catalog()), "SEMANTIC_WEIGHT", 0.6);
    expect(form.fields.find((f) => f.name === "SEMANTIC_WEIGHT")?.value).toBe(0.6);
    expect(form.fields.find((f) => f.name === "BM25_WEIGHT")?.value).toBe(0.4);
    // Both go in the patch, so the server's sum-to-1 validator passes.
    expect(patchBody(form)).toEqual({ SEMANTIC_WEIGHT: 0.6, BM25_WEIGHT: 0.4 });
  });

  it("the weight complement avoids float junk", () => {
    const form = setValue(initForm(catalog()), "BM25_WEIGHT", 0.3);
    expect(form.fields.find((f) => f.name === "SEMANTIC_WEIGHT")?.value).toBe(0.7);
  });
});

describe("patchBody", () => {
  it("contains only the dirty fields", () => {
    let form = initForm(catalog());
    form = setValue(form, "TOP_K", 25);
    form = setValue(form, "RETRIEVAL_METHOD", "reranked");
    expect(patchBody(form)).toEqual({ TOP_K: 25, RETRIEVAL_METHOD: "reranked" });
  });

  it("is empty when nothing is staged", () => {
    expect(patchBody(initForm(catalog()))).toEqual({});
  });
});

describe("willFlagStale", () => {
  it("is false for query-time-only edits", () => {
    const form = setValue(initForm(catalog()), "TOP_K", 25);
    expect(willFlagStale(form)).toBe(false);
  });

  it("is true when an ingest-time knob is staged", () => {
    const form = setValue(initForm(catalog()), "CHUNKING_STRATEGY", "token_based");
    expect(willFlagStale(form)).toBe(true);
  });

  it("is true for a staged CONTEXTUALIZE flip (boolean knob)", () => {
    const form = setValue(initForm(catalog()), "CONTEXTUALIZE", false);
    expect(willFlagStale(form)).toBe(true);
  });
});

describe("groupFields", () => {
  it("yields the spec §4.7 drawer order with only populated groups", () => {
    const groups = groupFields(initForm(catalog()));
    expect(groups.map(([name]) => name)).toEqual(["retrieval", "generation", "ingestion"]);
    expect(groups[0][1].map((f) => f.name)).toContain("RETRIEVAL_METHOD");
    expect(groups[1][1].map((f) => f.name)).toEqual(["LLM_TEMPERATURE"]);
  });
});
