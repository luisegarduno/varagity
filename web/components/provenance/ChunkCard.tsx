"use client";

import { useQuery } from "@tanstack/react-query";
import { ChevronRightIcon, ScanTextIcon } from "lucide-react";
import { useState } from "react";

import { PagePreview } from "@/components/provenance/PagePreview";
import { RankBadges } from "@/components/provenance/RankBadges";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsiblePanel,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import type { EvidenceChunk } from "@/lib/evidence";
import { highlightTerms } from "@/lib/highlight";
import { previewEligible } from "@/lib/preview";
import { configQuery } from "@/lib/queries";
import { formatScore } from "@/lib/trace";
import { cn } from "@/lib/utils";

/** DOM id of one chunk's card — the citation chips' scroll target. */
export function chunkCardId(chunkKey: string): string {
  return `evidence-${chunkKey}`;
}

/**
 * Chunk text with the query's terms marked (the base layer styles the
 * bare `mark`). Exported for {@link PagePreview}'s full-text fallback, so
 * a degraded preview renders byte-identically to an ineligible chunk.
 */
export function HighlightedText({
  text,
  query,
}: {
  text: string;
  query: string | null;
}) {
  return (
    <>
      {highlightTerms(text, query).map((segment, index) =>
        segment.highlighted ? (
          // Unstyled on purpose: the base layer's accent-tinted `mark`.
          <mark key={index}>{segment.text}</mark>
        ) : (
          <span key={index}>{segment.text}</span>
        ),
      )}
    </>
  );
}

/**
 * One evidence row (spec_v2 §4.6): rank + final score, the trace badges,
 * source provenance (file, page, format, OCR fallback), the contextual
 * blurb, and the expandable full chunk text with query-term highlights.
 * `className`/`style` pass through for the panel's arrival stagger.
 */
export function ChunkCard({
  chunk,
  query,
  className,
  style,
}: {
  chunk: EvidenceChunk;
  query: string | null;
  className?: string;
  style?: React.CSSProperties;
}) {
  const [expanded, setExpanded] = useState(false);
  const { data: config = null } = useQuery(configQuery());
  const ocr = chunk.extraction === "ocr_fallback";
  // Belt and braces with the server's kill switch: while config is still
  // in flight the chunk stays optimistic — a disabled server would answer
  // the locate with `preview_disabled` and the panel falls back to text.
  const preview = previewEligible(chunk) && config?.preview_enabled !== false;

  return (
    <article
      id={chunkCardId(chunk.key)}
      data-chunk-key={chunk.key}
      style={style}
      className={cn(
        "space-y-2 rounded-lg border border-border bg-card p-3 text-card-foreground",
        className,
      )}
    >
      <header className="flex items-baseline justify-between gap-2">
        <span
          className="font-mono text-sm font-semibold tabular-nums"
          title={chunk.key}
        >
          #{chunk.rank}
        </span>
        {chunk.score !== null && (
          <span
            className="font-mono text-xs text-muted-foreground tabular-nums"
            title={
              chunk.trace?.rerank_score !== null &&
              chunk.trace?.rerank_score !== undefined
                ? "Cross-encoder relevance (post-rerank)"
                : "Retrieval score"
            }
          >
            score {formatScore(chunk.score, 4)}
          </span>
        )}
      </header>

      {chunk.trace && <RankBadges trace={chunk.trace} />}

      <p className="flex flex-wrap items-center gap-x-1.5 gap-y-1 text-xs text-muted-foreground">
        <span
          className="max-w-full truncate font-medium text-foreground"
          title={chunk.source ?? undefined}
        >
          {chunk.fileName ?? chunk.source ?? "unknown source"}
        </span>
        {chunk.page !== null && <span>page {chunk.page}</span>}
        {chunk.fileType && (
          <Badge variant="outline" className="font-mono uppercase">
            {chunk.fileType}
          </Badge>
        )}
        {ocr && (
          <Badge
            variant="warning"
            className="font-mono uppercase"
            title="Extracted via OCR fallback — this text came off a scanned page"
          >
            <ScanTextIcon aria-hidden />
            OCR
          </Badge>
        )}
      </p>

      {chunk.context && (
        <p className="border-l-2 border-primary/25 pl-2 text-xs leading-relaxed text-muted-foreground italic">
          {chunk.context}
        </p>
      )}

      <Collapsible open={expanded} onOpenChange={setExpanded}>
        <CollapsibleTrigger
          render={
            <Button
              variant="ghost"
              size="xs"
              className="-ml-1 text-muted-foreground"
            />
          }
        >
          <ChevronRightIcon
            aria-hidden
            className="motion-safe:transition-transform group-aria-expanded/button:rotate-90"
          />
          {preview
            ? expanded
              ? "Hide preview"
              : "Show preview"
            : expanded
              ? "Hide full text"
              : "Show full text"}
        </CollapsibleTrigger>
        <CollapsiblePanel>
          {preview ? (
            // Mounts on expand (closed panels unmount), driving the locate.
            <PagePreview chunk={chunk} query={query} />
          ) : (
            <p className="mt-1 rounded-md bg-muted/50 p-2 text-xs leading-relaxed whitespace-pre-wrap">
              <HighlightedText text={chunk.content} query={query} />
            </p>
          )}
        </CollapsiblePanel>
      </Collapsible>
    </article>
  );
}
