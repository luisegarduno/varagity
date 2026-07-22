import { describe, expect, it } from "vitest";

import { previewPageUrl } from "@/lib/api";
import type { EvidenceChunk } from "@/lib/evidence";
import { cssRects, previewEligible, previewFallbackLabel } from "@/lib/preview";

/** A previewable baseline chunk (digital PDF); cases override from here. */
function chunk(overrides: Partial<EvidenceChunk> = {}): EvidenceChunk {
  return {
    key: "a398491c7441925f::0",
    docId: "a398491c7441925f",
    rank: 1,
    score: null,
    content: "The observatory's saltwater intake was rebuilt in 2019.",
    context: null,
    source: "/app/docs/reports/saltmere_observatory.pdf",
    fileName: "saltmere_observatory.pdf",
    fileType: "pdf",
    page: 1,
    extraction: "text",
    fileCreatedAt: null,
    fileModifiedAt: null,
    trace: null,
    ...overrides,
  };
}

describe("previewEligible", () => {
  it("accepts a digital PDF", () => {
    expect(previewEligible(chunk())).toBe(true);
  });

  it("rejects a scanned PDF — OCR text never matches the page image", () => {
    expect(previewEligible(chunk({ extraction: "ocr_fallback" }))).toBe(false);
  });

  it("accepts pptx regardless of extraction (office parses are always text)", () => {
    expect(previewEligible(chunk({ fileType: "pptx" }))).toBe(true);
    expect(
      previewEligible(chunk({ fileType: "pptx", extraction: "ocr_fallback" })),
    ).toBe(true);
  });

  it("rejects every other format", () => {
    for (const fileType of ["docx", "xlsx", "md", "html", "txt", null]) {
      expect(previewEligible(chunk({ fileType }))).toBe(false);
    }
  });

  it("rejects chunks with no resolvable document or source", () => {
    expect(previewEligible(chunk({ docId: null }))).toBe(false);
    expect(previewEligible(chunk({ source: null }))).toBe(false);
  });
});

describe("cssRects", () => {
  it("converts normalized rects to percentage offsets", () => {
    expect(cssRects([{ x0: 0.1, y0: 0.2, x1: 0.5, y1: 0.25 }])).toEqual([
      { left: "10%", top: "20%", width: "40%", height: "5%" },
    ]);
  });

  it("keeps sub-percent precision without float noise", () => {
    expect(cssRects([{ x0: 0, y0: 0, x1: 1 / 3, y1: 1 }])).toEqual([
      { left: "0%", top: "0%", width: "33.3333%", height: "100%" },
    ]);
  });

  it("clamps degenerate and out-of-range rects to the page", () => {
    expect(cssRects([{ x0: 0.6, y0: 0.9, x1: 0.4, y1: 1.4 }])).toEqual([
      { left: "60%", top: "90%", width: "0%", height: "10%" },
    ]);
    expect(cssRects([{ x0: -0.5, y0: -1, x1: 0.5, y1: 0.5 }])).toEqual([
      { left: "0%", top: "0%", width: "50%", height: "50%" },
    ]);
  });

  it("passes an empty rect list through", () => {
    expect(cssRects([])).toEqual([]);
  });
});

describe("previewFallbackLabel", () => {
  it("names the server's reason, humanized", () => {
    expect(previewFallbackLabel("file_changed")).toBe(
      "preview unavailable (file changed) — showing text",
    );
    expect(previewFallbackLabel("no_match")).toBe(
      "preview unavailable (no match) — showing text",
    );
  });

  it("stays generic when the server never got to say", () => {
    expect(previewFallbackLabel(null)).toBe(
      "preview unavailable — showing text",
    );
  });
});

describe("previewPageUrl", () => {
  it("builds the immutable page-image URL (the browser caches it)", () => {
    expect(previewPageUrl("a398491c7441925f", 2)).toBe(
      "http://localhost:8000/api/documents/a398491c7441925f/preview/page/2",
    );
  });
});
