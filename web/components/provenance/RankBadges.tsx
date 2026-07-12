import type { RetrievalTrace } from "@/lib/api";
import { buildTraceBadges, type BadgeTone } from "@/lib/trace";
import { cn } from "@/lib/utils";

const TONE_CLASSES: Record<BadgeTone, string> = {
  neutral: "bg-muted text-muted-foreground",
  muted: "bg-muted text-muted-foreground italic",
  up: "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400",
  down: "bg-rose-500/10 text-rose-600 dark:text-rose-400",
};

/**
 * The "why it ranked here" badge row: `sem #1 · bm25 #3 · fused 0.94 ·
 * rerank +2`, or `semantic-only` / `bm25-only` when an arm missed the
 * chunk (spec_v2 §4.6). Hover for the underlying scores.
 */
export function RankBadges({ trace }: { trace: RetrievalTrace }) {
  const badges = buildTraceBadges(trace);
  return (
    <span className="flex flex-wrap items-center gap-1">
      {badges.map((badge) => (
        <span
          key={badge.kind}
          title={badge.detail}
          data-kind={badge.kind}
          className={cn(
            "rounded border border-border/50 px-1.5 py-px font-mono text-[11px] leading-4 whitespace-nowrap",
            TONE_CLASSES[badge.tone],
          )}
        >
          {badge.label}
        </span>
      ))}
    </span>
  );
}
