"use client";

import { useEffect } from "react";

import { ChunkCard, chunkCardId } from "@/components/provenance/ChunkCard";
import type { Evidence } from "@/lib/evidence";
import { cn } from "@/lib/utils";

/** A citation chip's scroll request: bump `nonce` to re-trigger. */
export interface EvidenceScrollTarget {
  chunkKey: string;
  nonce: number;
}

// Preferred stage order; anything else the API adds renders after.
const STAGE_ORDER = ["retrieval", "generation", "total"];

function formatMs(ms: number): string {
  return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`;
}

function latencyLine(latencyMs: Record<string, number>): string {
  const stages = [
    ...STAGE_ORDER.filter((stage) => stage in latencyMs),
    ...Object.keys(latencyMs).filter((stage) => !STAGE_ORDER.includes(stage)),
  ];
  return stages
    .map((stage) => `${stage} ${formatMs(latencyMs[stage])}`)
    .join(" · ");
}

function retrievalLine(evidence: Evidence): string {
  const parts: string[] = [];
  if (evidence.method) parts.push(evidence.method);
  if (evidence.topK !== null) {
    parts.push(
      evidence.rerankedTo !== null
        ? `top_k ${evidence.topK} → reranked to ${evidence.rerankedTo}`
        : `top_k ${evidence.topK}`,
    );
  }
  parts.push(`${evidence.chunks.length} chunk${evidence.chunks.length === 1 ? "" : "s"}`);
  return parts.join(" · ");
}

/**
 * ★ "How this answer was built" (spec_v2 §4.6): the answer's evidence
 * rows — rank, score, trace badges, provenance, blurb, expandable text —
 * with the answer-level meta (method, top_k → reranked-to, per-stage
 * latency) around them. A plain right rail for now; the responsive
 * collapsible treatment is Phase 9.
 */
export function EvidencePanel({
  evidence,
  scrollTarget,
  className,
}: {
  evidence: Evidence | null;
  scrollTarget: EvidenceScrollTarget | null;
  className?: string;
}) {
  // Citation-chip clicks land here after the panel re-rendered with the
  // right message's evidence: scroll the card into view and pulse it.
  useEffect(() => {
    if (!scrollTarget) return;
    const card = document.getElementById(chunkCardId(scrollTarget.chunkKey));
    if (!card) return;
    card.scrollIntoView({ behavior: "smooth", block: "center" });
    card.classList.add("evidence-pulse");
    const settle = () => card.classList.remove("evidence-pulse");
    card.addEventListener("animationend", settle, { once: true });
    return () => {
      card.removeEventListener("animationend", settle);
      settle();
    };
  }, [scrollTarget]);

  return (
    <aside
      aria-label="How this answer was built"
      className={cn("flex-col bg-background", className)}
    >
      <header className="border-b border-border p-4">
        <h2 className="text-sm font-semibold">How this answer was built</h2>
        {evidence && (
          <p className="mt-1 font-mono text-xs text-muted-foreground">
            {retrievalLine(evidence)}
          </p>
        )}
      </header>

      <div className="min-h-0 flex-1 space-y-2 overflow-y-auto p-3">
        {evidence === null ? (
          <p className="p-4 text-center text-sm text-muted-foreground">
            Ask a question and the retrieved evidence — with each chunk&apos;s
            ranks, scores, and context — appears here.
          </p>
        ) : evidence.chunks.length === 0 ? (
          <p className="p-4 text-center text-sm text-muted-foreground">
            No chunks were retrieved for this answer.
          </p>
        ) : (
          evidence.chunks.map((chunk) => (
            <ChunkCard key={chunk.key} chunk={chunk} query={evidence.query} />
          ))
        )}
      </div>

      {evidence?.latencyMs && (
        <footer className="border-t border-border p-3">
          <p className="font-mono text-xs text-muted-foreground">
            {latencyLine(evidence.latencyMs)}
          </p>
        </footer>
      )}
    </aside>
  );
}
