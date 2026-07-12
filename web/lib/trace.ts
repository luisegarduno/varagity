/**
 * Trace-badge building: why one chunk ranked where it did (spec_v2 §4.6).
 *
 * The pure logic behind `RankBadges` — the browser counterpart of the
 * CLI's `trace_badges` (`varagity/debug/show.py`), kept format-compatible
 * so the panel's numbers match a `-v 2` terminal run: ranks as `#N`,
 * fused score to two decimals, rerank delta always signed. Where the CLI
 * prints `sem —` for an arm that never surfaced the chunk, the panel
 * follows the spec's richer vocabulary: a single `semantic-only` /
 * `bm25-only` badge.
 */
import type { RetrievalTrace } from "@/lib/api";

/** Visual tone of one badge (mapped to classes by `RankBadges`). */
export type BadgeTone = "neutral" | "muted" | "up" | "down";

/** One rendered badge: a short label plus a hover detail. */
export interface TraceBadge {
  /** Stable kind (React key + test hook). */
  kind:
    | "semantic"
    | "bm25"
    | "semantic-only"
    | "bm25-only"
    | "fused"
    | "rerank";
  /** The visible text, e.g. `sem #1` or `rerank +2`. */
  label: string;
  /** Visual tone. */
  tone: BadgeTone;
  /** Hover detail (the underlying score), when there is one. */
  detail?: string;
}

/** Format a score for display (badge labels use two decimals). */
export function formatScore(score: number, digits = 2): string {
  return score.toFixed(digits);
}

/** Format a rerank delta with an explicit sign (`+2`, `-1`, `+0`). */
export function formatDelta(delta: number): string {
  return delta >= 0 ? `+${delta}` : `${delta}`;
}

/**
 * Build the badge list for one chunk's retrieval trace.
 *
 * Both arms present → `sem #a · bm25 #b`; one arm absent → the spec's
 * `semantic-only` / `bm25-only` marker instead. The fused badge always
 * renders; the rerank badge only when the rerank stage ran
 * (`rerank_delta` present), signed and toned by direction.
 */
export function buildTraceBadges(trace: RetrievalTrace): TraceBadge[] {
  const badges: TraceBadge[] = [];
  const semantic = trace.semantic_rank;
  const bm25 = trace.bm25_rank;

  if (semantic !== null && bm25 !== null) {
    badges.push({
      kind: "semantic",
      label: `sem #${semantic}`,
      tone: "neutral",
      detail:
        trace.semantic_score !== null
          ? `semantic cosine ${formatScore(trace.semantic_score, 4)}`
          : undefined,
    });
    badges.push({
      kind: "bm25",
      label: `bm25 #${bm25}`,
      tone: "neutral",
      detail:
        trace.bm25_score !== null
          ? `bm25 relevance ${formatScore(trace.bm25_score, 2)}`
          : undefined,
    });
  } else if (semantic !== null) {
    badges.push({
      kind: "semantic-only",
      label: `semantic-only #${semantic}`,
      tone: "muted",
      detail:
        trace.semantic_score !== null
          ? `only the semantic arm surfaced this chunk (cosine ${formatScore(trace.semantic_score, 4)})`
          : "only the semantic arm surfaced this chunk",
    });
  } else if (bm25 !== null) {
    badges.push({
      kind: "bm25-only",
      label: `bm25-only #${bm25}`,
      tone: "muted",
      detail:
        trace.bm25_score !== null
          ? `only the BM25 arm surfaced this chunk (relevance ${formatScore(trace.bm25_score, 2)})`
          : "only the BM25 arm surfaced this chunk",
    });
  }

  badges.push({
    kind: "fused",
    label: `fused ${formatScore(trace.fused_score)}`,
    tone: "neutral",
    detail: `rank #${trace.fused_rank} after fusion`,
  });

  if (trace.rerank_delta !== null) {
    badges.push({
      kind: "rerank",
      label: `rerank ${formatDelta(trace.rerank_delta)}`,
      tone:
        trace.rerank_delta > 0 ? "up" : trace.rerank_delta < 0 ? "down" : "neutral",
      detail:
        trace.rerank_score !== null
          ? `cross-encoder relevance ${formatScore(trace.rerank_score, 4)}; moved ${describeDelta(trace.rerank_delta)}`
          : `moved ${describeDelta(trace.rerank_delta)}`,
    });
  }

  return badges;
}

function describeDelta(delta: number): string {
  if (delta > 0) return `up ${delta} place${delta === 1 ? "" : "s"}`;
  if (delta < 0) return `down ${-delta} place${delta === -1 ? "" : "s"}`;
  return "nowhere (rank unchanged)";
}
