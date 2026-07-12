"use client";

import { ChevronDownIcon, ChevronRightIcon, ScanTextIcon } from "lucide-react";
import { useState } from "react";

import { RankBadges } from "@/components/provenance/RankBadges";
import { Button } from "@/components/ui/button";
import type { EvidenceChunk } from "@/lib/evidence";
import { highlightTerms } from "@/lib/highlight";
import { formatScore } from "@/lib/trace";
import { cn } from "@/lib/utils";

/** DOM id of one chunk's card — the citation chips' scroll target. */
export function chunkCardId(chunkKey: string): string {
  return `evidence-${chunkKey}`;
}

function HighlightedText({
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
          <mark
            key={index}
            className="rounded-sm bg-amber-200/70 px-0.5 text-inherit dark:bg-amber-500/30"
          >
            {segment.text}
          </mark>
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
 */
export function ChunkCard({
  chunk,
  query,
}: {
  chunk: EvidenceChunk;
  query: string | null;
}) {
  const [expanded, setExpanded] = useState(false);
  const ocr = chunk.extraction === "ocr_fallback";

  return (
    <article
      id={chunkCardId(chunk.key)}
      data-chunk-key={chunk.key}
      className="space-y-2 rounded-lg border border-border bg-card p-3 text-card-foreground"
    >
      <header className="flex items-baseline justify-between gap-2">
        <span className="text-sm font-semibold" title={chunk.key}>
          #{chunk.rank}
        </span>
        {chunk.score !== null && (
          <span
            className="font-mono text-xs text-muted-foreground"
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
          <span className="rounded border border-border/50 bg-muted px-1 py-px font-mono text-[10px] uppercase">
            {chunk.fileType}
          </span>
        )}
        {ocr && (
          <span
            className="inline-flex items-center gap-0.5 rounded border border-amber-500/40 bg-amber-500/10 px-1 py-px font-mono text-[10px] text-amber-700 uppercase dark:text-amber-400"
            title="Extracted via OCR fallback — this text came off a scanned page"
          >
            <ScanTextIcon className="size-3" aria-hidden />
            OCR
          </span>
        )}
      </p>

      {chunk.context && (
        <p className="border-l-2 border-border pl-2 text-xs text-muted-foreground italic">
          {chunk.context}
        </p>
      )}

      <div>
        <Button
          variant="ghost"
          size="xs"
          className="-ml-1 text-muted-foreground"
          aria-expanded={expanded}
          onClick={() => setExpanded((open) => !open)}
        >
          {expanded ? (
            <ChevronDownIcon aria-hidden />
          ) : (
            <ChevronRightIcon aria-hidden />
          )}
          {expanded ? "Hide full text" : "Show full text"}
        </Button>
        {expanded && (
          <p
            className={cn(
              "mt-1 rounded-md bg-muted/50 p-2 text-xs leading-relaxed",
              "whitespace-pre-wrap",
            )}
          >
            <HighlightedText text={chunk.content} query={query} />
          </p>
        )}
      </div>
    </article>
  );
}
