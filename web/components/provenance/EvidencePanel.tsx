"use client";

import { PanelRightCloseIcon } from "lucide-react";
import { useRef, type RefObject } from "react";

import { ChunkCard } from "@/components/provenance/ChunkCard";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Drawer,
  DrawerContent,
  DrawerHeader,
  DrawerTitle,
} from "@/components/ui/drawer";
import { useMountEffect } from "@/hooks/use-mount-effect";
import {
  formatTokensPerSecond,
  LIVE_EVIDENCE_KEY,
  type Evidence,
  type EvidenceUsage,
} from "@/lib/evidence";
import { cn } from "@/lib/utils";

/** A citation chip's scroll request: bump `nonce` to re-trigger. */
export interface EvidenceScrollTarget {
  chunkKey: string;
  nonce: number;
}

/**
 * Scrolls one evidence card into view and pulses it. The list keys this on
 * the request's `nonce`, so each citation click mounts a fresh instance and
 * replays the scroll — which is what the nonce existed for all along.
 * Unmounting (the sheet closing, the request clearing) settles the pulse.
 */
function ScrollToChunk({
  chunkKey,
  listRef,
}: {
  chunkKey: string;
  listRef: RefObject<HTMLDivElement | null>;
}) {
  useMountEffect(() => {
    const card = listRef.current?.querySelector<HTMLElement>(
      `[data-chunk-key="${CSS.escape(chunkKey)}"]`,
    );
    if (!card) return;
    card.scrollIntoView({ behavior: "smooth", block: "center" });
    card.classList.add("evidence-pulse");
    const settle = () => card.classList.remove("evidence-pulse");
    card.addEventListener("animationend", settle, { once: true });
    return () => {
      card.removeEventListener("animationend", settle);
      settle();
    };
  });
  return null;
}

// Preferred stage order; anything else the API adds renders after.
const STAGE_ORDER = ["retrieval", "generation", "total"];

// One unit for the whole footer: seconds, two decimals.
function formatSeconds(ms: number): string {
  return `${(ms / 1000).toFixed(2)} s`;
}

function latencyLine(latencyMs: Record<string, number>): string {
  const stages = [
    ...STAGE_ORDER.filter((stage) => stage in latencyMs),
    ...Object.keys(latencyMs).filter((stage) => !STAGE_ORDER.includes(stage)),
  ];
  return stages
    .map((stage) => `${stage} ${formatSeconds(latencyMs[stage])}`)
    .join(" · ");
}

/**
 * The token-accounting line: counts plus the model server's own decode
 * rate. Segments drop out individually when unreported (only llama.cpp
 * reports a rate), and the whole line disappears for turns answered
 * before this page load — usage is session-only, never persisted.
 */
function usageLine(usage: EvidenceUsage): string | null {
  const segments: string[] = [];
  if (usage.promptTokens !== null) segments.push(`${usage.promptTokens} prompt`);
  if (usage.completionTokens !== null)
    segments.push(`${usage.completionTokens} completion`);
  if (usage.tokensPerSecond !== null)
    segments.push(formatTokensPerSecond(usage.tokensPerSecond));
  return segments.length > 0 ? segments.join(" · ") : null;
}

/** The footer stat lines both hosts share: per-stage latency, then usage. */
function FooterStats({ evidence }: { evidence: Evidence }) {
  const lines = [
    evidence.latencyMs && latencyLine(evidence.latencyMs),
    evidence.usage && usageLine(evidence.usage),
  ].filter((line): line is string => Boolean(line));
  return (
    <>
      {lines.map((line) => (
        <p
          key={line}
          className="font-mono text-xs text-muted-foreground tabular-nums"
        >
          {line}
        </p>
      ))}
    </>
  );
}

/** Whether {@link FooterStats} would render anything for this evidence. */
function hasFooterStats(evidence: Evidence | null): evidence is Evidence {
  return Boolean(
    evidence && (evidence.latencyMs || (evidence.usage && usageLine(evidence.usage))),
  );
}

/**
 * The answer-level meta: method badge, top_k → reranked-to, count — plus,
 * when the chat engine rewrote the turn, the "Searched for: …" line
 * (spec_v3 §4.7). Same "how this answer was built" promise: if retrieval
 * ran on something other than what was typed, the reader gets to see it.
 */
function EvidenceMeta({ evidence }: { evidence: Evidence }) {
  const counts: string[] = [];
  if (evidence.topK !== null) {
    counts.push(
      evidence.rerankedTo !== null
        ? `top_k ${evidence.topK} → ${evidence.rerankedTo}`
        : `top_k ${evidence.topK}`,
    );
  }
  counts.push(
    `${evidence.chunks.length} chunk${evidence.chunks.length === 1 ? "" : "s"}`,
  );
  return (
    <>
      <p className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted-foreground">
        {evidence.method && (
          <Badge variant="accent" className="font-mono">
            {evidence.method}
          </Badge>
        )}
        <span className="font-mono tabular-nums">{counts.join(" · ")}</span>
      </p>
      {evidence.condensedQuery !== null && (
        <p className="text-xs text-muted-foreground">
          Searched for:{" "}
          <span className="text-foreground/80 italic">
            {evidence.condensedQuery}
          </span>
        </p>
      )}
    </>
  );
}

/**
 * The evidence rows themselves, shared by the desktop rail and the mobile
 * sheet. Owns the citation-click scroll+pulse (scoped to this list, so the
 * two hosts never fight over card ids) and the arrival stagger: fresh
 * evidence rises in card by card; pinned or reloaded answers render at
 * rest (the animation is keyed to the live evidence key, not the render).
 */
function EvidenceCardList({
  evidence,
  scrollTarget,
  className,
}: {
  evidence: Evidence | null;
  scrollTarget: EvidenceScrollTarget | null;
  className?: string;
}) {
  const listRef = useRef<HTMLDivElement | null>(null);

  const live = evidence?.key === LIVE_EVIDENCE_KEY;

  return (
    <div ref={listRef} className={cn("space-y-2", className)}>
      {/* Citation-chip clicks land here after the host re-rendered with the
          right answer's evidence. */}
      {scrollTarget && (
        <ScrollToChunk
          key={scrollTarget.nonce}
          chunkKey={scrollTarget.chunkKey}
          listRef={listRef}
        />
      )}
      {evidence === null ? (
        <p className="p-4 text-center text-sm text-muted-foreground">
          Evidence appears here as answers are built.
        </p>
      ) : evidence.chunks.length === 0 ? (
        <p className="p-4 text-center text-sm text-muted-foreground">
          No chunks were retrieved for this answer.
        </p>
      ) : (
        evidence.chunks.map((chunk, index) => (
          <ChunkCard
            key={chunk.key}
            chunk={chunk}
            query={evidence.query}
            className={
              live
                ? "motion-safe:animate-in motion-safe:fade-in motion-safe:slide-in-from-bottom-2 motion-safe:fill-mode-backwards motion-safe:duration-300"
                : undefined
            }
            // Stagger caps at eight cards so a deep list doesn't dawdle.
            style={
              live
                ? { animationDelay: `${Math.min(index, 8) * 40}ms` }
                : undefined
            }
          />
        ))
      )}
    </div>
  );
}

/**
 * ★ "How this answer was built" (spec_v2 §4.6): the answer's evidence
 * rows — rank, score, trace badges, provenance, blurb, expandable text —
 * with the answer-level meta (method, top_k → reranked-to, per-stage
 * latency) around them. This is the desktop rail host; the same content
 * renders in the mobile bottom sheet via {@link EvidenceSheet}.
 */
export function EvidencePanel({
  evidence,
  scrollTarget,
  className,
  onClose,
}: {
  evidence: Evidence | null;
  scrollTarget: EvidenceScrollTarget | null;
  className?: string;
  /** Renders a close affordance in the header (the collapsible rail). */
  onClose?: () => void;
}) {
  return (
    <aside
      aria-label="How this answer was built"
      className={cn("flex-col bg-background", className)}
    >
      <header className="flex items-start justify-between gap-2 border-b border-border p-4">
        <div className="min-w-0 space-y-1.5">
          <h2 className="font-heading text-base leading-snug font-normal italic">
            How this answer was built
          </h2>
          {evidence && <EvidenceMeta evidence={evidence} />}
        </div>
        {onClose && (
          <Button
            variant="ghost"
            size="icon-xs"
            aria-label="Hide evidence panel"
            title="Hide evidence panel"
            className="-mt-1 -mr-1.5 text-muted-foreground"
            onClick={onClose}
          >
            <PanelRightCloseIcon />
          </Button>
        )}
      </header>

      <EvidenceCardList
        evidence={evidence}
        scrollTarget={scrollTarget}
        className="min-h-0 flex-1 overflow-y-auto p-3 scroll-fade-y"
      />

      {hasFooterStats(evidence) && (
        <footer className="space-y-0.5 border-t border-border px-4 py-2.5">
          <FooterStats evidence={evidence} />
        </footer>
      )}
    </aside>
  );
}

/**
 * The narrow-screen host: the same evidence content inside the bottom
 * sheet (spec_v2 §4.8). Controlled by the conversation — citation chips
 * and the per-answer sources affordance open it below `lg`. The sheet
 * itself scrolls; swipe down (or Esc) dismisses.
 */
export function EvidenceSheet({
  evidence,
  scrollTarget,
  open,
  onOpenChange,
}: {
  evidence: Evidence | null;
  scrollTarget: EvidenceScrollTarget | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  return (
    <Drawer side="bottom" open={open} onOpenChange={onOpenChange}>
      <DrawerContent>
        <DrawerHeader>
          <DrawerTitle className="italic">
            How this answer was built
          </DrawerTitle>
          {evidence && <EvidenceMeta evidence={evidence} />}
        </DrawerHeader>
        <EvidenceCardList evidence={evidence} scrollTarget={scrollTarget} />
        {hasFooterStats(evidence) && (
          <div className="space-y-0.5 border-t border-border pt-3">
            <FooterStats evidence={evidence} />
          </div>
        )}
      </DrawerContent>
    </Drawer>
  );
}
