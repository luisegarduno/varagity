/**
 * Pure logic behind the evidence panel's page preview (ADR-010): which
 * chunks get the "Show preview" affordance, and how the server's
 * normalized highlight rects become CSS. The fetch/render side lives in
 * `lib/queries.ts` + `components/provenance/PagePreview.tsx`.
 */
import type { PreviewRect } from "@/lib/api";
import type { EvidenceChunk } from "@/lib/evidence";

/**
 * Whether a chunk's source can be previewed at all: digital PDFs and PPTX
 * decks with a resolvable document. Scanned PDFs (`ocr_fallback`) keep
 * the full-text view — their text never matches the page image — and so
 * does everything else. This is a client-side gate only; the server
 * re-checks and answers `available:false` if the two ever disagree.
 */
export function previewEligible(chunk: EvidenceChunk): boolean {
  if (!chunk.docId || !chunk.source) return false;
  if (chunk.fileType === "pptx") return true;
  return chunk.fileType === "pdf" && chunk.extraction !== "ocr_fallback";
}

/** One highlight rectangle as CSS percentage offsets of the page image. */
export interface CssRect {
  left: string;
  top: string;
  width: string;
  height: string;
}

/** `0.4 → "40%"`, trimmed to at most four decimals so tests stay exact. */
function pct(fraction: number): string {
  return `${Number((fraction * 100).toFixed(4))}%`;
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

/**
 * Convert the server's normalized rects (`[0, 1]`, top-left origin — the
 * y-flip already happened server-side) into absolute-positioning CSS.
 * Percentages scale with the rendered image, so the same rects overlay
 * the inline preview and the enlarge dialog. Out-of-range or inverted
 * rects clamp to the page rather than spilling over the card.
 */
export function cssRects(rects: PreviewRect[]): CssRect[] {
  return rects.map((rect) => {
    const left = clamp01(rect.x0);
    const top = clamp01(rect.y0);
    return {
      left: pct(left),
      top: pct(top),
      width: pct(Math.max(0, clamp01(rect.x1) - left)),
      height: pct(Math.max(0, clamp01(rect.y1) - top)),
    };
  });
}

/**
 * The muted one-liner under the full-text fallback, naming why the
 * preview degraded. `reason` is the server's snake_case condition
 * (`file_changed`, `no_match`, …); `null` covers transport errors, where
 * the server never got to say.
 */
export function previewFallbackLabel(reason: string | null): string {
  const cause = reason ? ` (${reason.replaceAll("_", " ")})` : "";
  return `preview unavailable${cause} — showing text`;
}
