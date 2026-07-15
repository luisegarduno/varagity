import { Badge } from "@/components/ui/badge";
import type { RetrievalTrace } from "@/lib/api";
import { buildTraceBadges, type BadgeTone } from "@/lib/trace";
import { cn } from "@/lib/utils";

/** lib/trace's tone vocabulary mapped onto the Badge variants. */
const TONE_VARIANTS: Record<
  BadgeTone,
  React.ComponentProps<typeof Badge>["variant"]
> = {
  neutral: "default",
  muted: "default",
  up: "success",
  down: "destructive",
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
        <Badge
          key={badge.kind}
          title={badge.detail}
          data-kind={badge.kind}
          variant={TONE_VARIANTS[badge.tone]}
          className={cn("font-mono", badge.tone === "muted" && "italic")}
        >
          {badge.label}
        </Badge>
      ))}
    </span>
  );
}
