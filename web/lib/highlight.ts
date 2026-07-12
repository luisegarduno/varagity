/**
 * Client-side query-term highlighting for expanded chunk text
 * (spec_v2 §4.6): a plain string highlight, consistent with the
 * deferred-document-preview decision — no ingest-time offsets involved.
 */

/** One run of chunk text, highlighted or not; runs concatenate to the input. */
export interface HighlightSegment {
  text: string;
  highlighted: boolean;
}

// Common short English function words that survive the length filter but
// only add noise as highlights.
const STOPWORDS = new Set([
  "and",
  "are",
  "about",
  "did",
  "does",
  "for",
  "from",
  "how",
  "into",
  "that",
  "the",
  "their",
  "there",
  "this",
  "was",
  "were",
  "what",
  "when",
  "where",
  "which",
  "who",
  "whom",
  "why",
  "with",
]);

function escapeRegExp(term: string): string {
  return term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * The query's highlightable terms: lowercased word tokens, minus
 * stopwords and tokens shorter than three characters, deduplicated and
 * sorted longest-first (so overlapping alternatives prefer the longest
 * match).
 */
export function queryTerms(query: string | null): string[] {
  if (!query) return [];
  const tokens = query
    .toLowerCase()
    .split(/[^\p{L}\p{N}]+/u)
    .filter((token) => token.length >= 3 && !STOPWORDS.has(token));
  return [...new Set(tokens)].sort((a, b) => b.length - a.length);
}

/**
 * Split `text` into segments, marking occurrences of the query's terms.
 *
 * Matching is case-insensitive and anchored on the left word boundary
 * only, so `kelp` also lights up in `kelps` (the plural's tail stays
 * unhighlighted). No query terms → one unhighlighted segment.
 */
export function highlightTerms(
  text: string,
  query: string | null,
): HighlightSegment[] {
  if (!text) return [];
  const terms = queryTerms(query);
  if (terms.length === 0) return [{ text, highlighted: false }];

  const pattern = new RegExp(
    `(?<![\\p{L}\\p{N}])(?:${terms.map(escapeRegExp).join("|")})`,
    "giu",
  );
  const segments: HighlightSegment[] = [];
  let cursor = 0;
  for (const match of text.matchAll(pattern)) {
    const start = match.index;
    if (start > cursor) {
      segments.push({ text: text.slice(cursor, start), highlighted: false });
    }
    segments.push({ text: match[0], highlighted: true });
    cursor = start + match[0].length;
  }
  if (cursor < text.length) {
    segments.push({ text: text.slice(cursor), highlighted: false });
  }
  return segments;
}
