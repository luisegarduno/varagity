import { describe, expect, it } from "vitest";

import {
  annotateCitations,
  CITATION_HREF_PREFIX,
  citationIdFromHref,
  matchSource,
  type CitationSourceRef,
} from "@/lib/citations";

const refs: CitationSourceRef[] = [
  { source: "/docs/marine/kelp_corridor.md", fileName: "kelp_corridor.md" },
  { source: "/docs/aurora_station.md", fileName: "aurora_station.md" },
  { source: "/docs/scans/survey.pdf", fileName: "survey.pdf" },
];

describe("annotateCitations", () => {
  it("extracts a `[SOURCE]: /path` marker into a chip link", () => {
    const { markdown, citations } = annotateCitations(
      "Kelp grows fast [SOURCE]: /docs/marine/kelp_corridor.md and spreads.",
      refs,
    );
    expect(citations).toHaveLength(1);
    expect(citations[0]).toMatchObject({
      id: 0,
      path: "/docs/marine/kelp_corridor.md",
      label: "kelp_corridor.md",
      chunkIndex: 0,
    });
    expect(markdown).toBe(
      `Kelp grows fast [kelp_corridor.md](${CITATION_HREF_PREFIX}0) and spreads.`,
    );
  });

  it("handles the prompt's own two-space spacing and bracket form", () => {
    const twoSpace = annotateCitations(
      "[SOURCE]:  /docs/aurora_station.md",
      refs,
    );
    expect(twoSpace.citations[0]?.chunkIndex).toBe(1);

    const bracketed = annotateCitations(
      "See [SOURCE: /docs/aurora_station.md] for details.",
      refs,
    );
    expect(bracketed.citations[0]?.chunkIndex).toBe(1);
    expect(bracketed.markdown).toBe(
      `See [aurora_station.md](${CITATION_HREF_PREFIX}0) for details.`,
    );
  });

  it("rescues a line-initial marker markdown would swallow as a definition", () => {
    // `[SOURCE]: /path` at line start is a CommonMark link-reference
    // definition — the rewrite must replace it so it stays visible.
    const { markdown } = annotateCitations(
      "The fact.\n\n[SOURCE]: /docs/aurora_station.md",
      refs,
    );
    expect(markdown).toContain(`[aurora_station.md](${CITATION_HREF_PREFIX}0)`);
    expect(markdown).not.toContain("[SOURCE]");
  });

  it("strips wrapping backticks/parens and trailing punctuation", () => {
    const backticked = annotateCitations(
      "Fact ([SOURCE]: `/docs/scans/survey.pdf`).",
      refs,
    );
    expect(backticked.citations[0]?.path).toBe("/docs/scans/survey.pdf");
    expect(backticked.citations[0]?.chunkIndex).toBe(2);

    const sentenceEnd = annotateCitations(
      "Fact from [SOURCE]: /docs/scans/survey.pdf.",
      refs,
    );
    expect(sentenceEnd.citations[0]?.path).toBe("/docs/scans/survey.pdf");
    expect(sentenceEnd.markdown).toBe(
      `Fact from [survey.pdf](${CITATION_HREF_PREFIX}0).`,
    );
  });

  it("flags a citation whose source is not in the evidence", () => {
    const { citations } = annotateCitations(
      "Claim [SOURCE]: /docs/never_retrieved.md here.",
      refs,
    );
    expect(citations[0]?.chunkIndex).toBeNull();
    expect(citations[0]?.label).toBe("never_retrieved.md");
  });

  it("numbers multiple citations in answer order", () => {
    const { markdown, citations } = annotateCitations(
      "A [SOURCE]: /docs/aurora_station.md then B [SOURCE]: /docs/scans/survey.pdf",
      refs,
    );
    expect(citations.map((c) => c.chunkIndex)).toEqual([1, 2]);
    expect(markdown).toContain(`${CITATION_HREF_PREFIX}0`);
    expect(markdown).toContain(`${CITATION_HREF_PREFIX}1`);
  });

  it("leaves prose mentions and bare markers untouched", () => {
    const bare = "The [SOURCE] above says so.";
    expect(annotateCitations(bare, refs).markdown).toBe(bare);
    expect(annotateCitations(bare, refs).citations).toHaveLength(0);

    const noPath = "[SOURCE]:";
    expect(annotateCitations(noPath, refs).markdown).toBe(noPath);
  });
});

describe("matchSource", () => {
  it("matches an exact source path", () => {
    expect(matchSource("/docs/marine/kelp_corridor.md", refs)).toBe(0);
  });

  it("matches a relative path as a segment suffix", () => {
    expect(matchSource("marine/kelp_corridor.md", refs)).toBe(0);
    expect(matchSource("kelp_corridor.md", refs)).toBe(0);
  });

  it("matches by basename, case-insensitively", () => {
    expect(matchSource("/other/root/SURVEY.PDF", refs)).toBe(2);
  });

  it("returns null when nothing matches", () => {
    expect(matchSource("/docs/unknown.txt", refs)).toBeNull();
  });

  it("prefers the best-ranked row when several share a file", () => {
    const dupes: CitationSourceRef[] = [
      { source: "/docs/a.md", fileName: "a.md" },
      { source: "/docs/a.md", fileName: "a.md" },
    ];
    expect(matchSource("/docs/a.md", dupes)).toBe(0);
  });
});

describe("citationIdFromHref", () => {
  it("round-trips the ids annotateCitations writes", () => {
    expect(citationIdFromHref(`${CITATION_HREF_PREFIX}0`)).toBe(0);
    expect(citationIdFromHref(`${CITATION_HREF_PREFIX}12`)).toBe(12);
  });

  it("rejects other hrefs", () => {
    expect(citationIdFromHref("https://example.com")).toBeNull();
    expect(citationIdFromHref("#some-anchor")).toBeNull();
    expect(citationIdFromHref(`${CITATION_HREF_PREFIX}nope`)).toBeNull();
    expect(citationIdFromHref(undefined)).toBeNull();
  });
});
