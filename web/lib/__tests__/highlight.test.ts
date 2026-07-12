import { describe, expect, it } from "vitest";

import { highlightTerms, queryTerms } from "@/lib/highlight";

function joined(segments: { text: string }[]): string {
  return segments.map((segment) => segment.text).join("");
}

function marked(segments: { text: string; highlighted: boolean }[]): string[] {
  return segments
    .filter((segment) => segment.highlighted)
    .map((segment) => segment.text);
}

describe("queryTerms", () => {
  it("keeps content words, dropping short tokens and stopwords", () => {
    expect(queryTerms("What is the kelp corridor of Aurora?")).toEqual(
      expect.arrayContaining(["kelp", "corridor", "aurora"]),
    );
    expect(queryTerms("What is the kelp?")).not.toContain("what");
    expect(queryTerms("What is the kelp?")).not.toContain("the");
    expect(queryTerms("is it an ok id")).toEqual([]);
  });

  it("deduplicates and sorts longest-first", () => {
    expect(queryTerms("kelp kelp corridors kelp")).toEqual([
      "corridors",
      "kelp",
    ]);
  });

  it("is empty for a null query", () => {
    expect(queryTerms(null)).toEqual([]);
  });
});

describe("highlightTerms", () => {
  it("segments cover the input text exactly", () => {
    const text = "The kelp corridor shelters juvenile fish year-round.";
    const segments = highlightTerms(text, "kelp corridor fish");
    expect(joined(segments)).toBe(text);
    expect(marked(segments)).toEqual(["kelp", "corridor", "fish"]);
  });

  it("matches case-insensitively and at word starts only", () => {
    const segments = highlightTerms(
      "Kelp beds; kelps everywhere. Helping is not kelp.",
      "kelp",
    );
    // "Helping" contains "elp" but not at a word start; "kelps" lights
    // its "kelp" prefix.
    expect(marked(segments)).toEqual(["Kelp", "kelp", "kelp"]);
    expect(joined(segments)).toContain("Helping");
  });

  it("returns one unhighlighted segment when the query has no terms", () => {
    expect(highlightTerms("Some chunk text.", "is a of")).toEqual([
      { text: "Some chunk text.", highlighted: false },
    ]);
    expect(highlightTerms("Some chunk text.", null)).toEqual([
      { text: "Some chunk text.", highlighted: false },
    ]);
  });

  it("escapes regex specials in query terms", () => {
    const segments = highlightTerms(
      "The cost is $4.20 (approx).",
      "what does $4.20 (approx) cost?",
    );
    expect(joined(segments)).toBe("The cost is $4.20 (approx).");
    expect(marked(segments)).toContain("cost");
  });

  it("returns no segments for empty text", () => {
    expect(highlightTerms("", "kelp")).toEqual([]);
  });
});
