"use client";

import { TriangleAlertIcon } from "lucide-react";
import type { Components } from "react-markdown";

import { citationIdFromHref, type Citation } from "@/lib/citations";
import { cn } from "@/lib/utils";

/**
 * One inline `[SOURCE]` citation chip. A matched chip carries its
 * evidence rank and clicks through to the panel's chunk card; a citation
 * whose source is *not* in the retrieved evidence renders as a warning —
 * the model may have strayed from its grounding (spec_v2 §4.6). While
 * the answer is still streaming, an unmatched chip stays neutral: the
 * path may simply not have finished arriving.
 */
export function CitationChip({
  citation,
  onCite,
  pending = false,
}: {
  citation: Citation;
  onCite: (chunkIndex: number) => void;
  pending?: boolean;
}) {
  const chunkIndex = citation.chunkIndex;
  if (chunkIndex === null) {
    if (pending) {
      return (
        <span
          data-citation="pending"
          className="inline-flex max-w-full items-center gap-1 rounded-md border border-border bg-muted px-1.5 align-baseline font-mono text-xs text-muted-foreground"
        >
          <span className="truncate">{citation.label}</span>
        </span>
      );
    }
    return (
      <span
        data-citation="missing"
        title={`Cited "${citation.path}", which is not in the retrieved evidence`}
        className="inline-flex max-w-full items-center gap-1 rounded-md border border-destructive/40 bg-destructive/10 px-1.5 align-baseline font-mono text-xs text-destructive"
      >
        <TriangleAlertIcon className="size-3 shrink-0" aria-hidden />
        <span className="truncate">{citation.label}</span>
        <span className="sr-only">(not in the retrieved evidence)</span>
      </span>
    );
  }
  return (
    <button
      type="button"
      data-citation="matched"
      title={`Show evidence #${chunkIndex + 1} — ${citation.path}`}
      onClick={() => onCite(chunkIndex)}
      className={cn(
        // The Badge accent recipe, interactive: the chip is the doorway
        // into the evidence panel, so it carries the accent quietly.
        "inline-flex max-w-full cursor-pointer items-center gap-1 rounded-md border border-primary/15 bg-primary/10 px-1.5 align-baseline font-mono text-xs text-primary",
        "dark:border-primary/25 dark:bg-primary/15 dark:text-[oklch(0.78_calc(var(--accent-chroma)*0.7)_var(--accent-hue))]",
        "transition-colors hover:bg-primary/15 focus-visible:ring-2 focus-visible:ring-ring/50 focus-visible:outline-none dark:hover:bg-primary/25",
      )}
    >
      <span className="opacity-70 tabular-nums">#{chunkIndex + 1}</span>
      <span className="truncate">{citation.label}</span>
    </button>
  );
}

/**
 * A react-markdown `components` override that renders the links
 * {@link annotateCitations} wrote (`#varagity-cite-N`) as
 * {@link CitationChip}s, and any other link as a plain anchor.
 * `pending` marks a still-streaming answer (see {@link CitationChip}).
 */
export function citationComponents(
  citations: readonly Citation[],
  onCite: (chunkIndex: number) => void,
  pending = false,
): Components {
  function CitationAnchor({
    href,
    children,
  }: {
    href?: string;
    children?: React.ReactNode;
  }) {
    const id = citationIdFromHref(href);
    const citation = id !== null ? citations[id] : undefined;
    if (!citation) {
      return (
        <a href={href} target="_blank" rel="noreferrer">
          {children}
        </a>
      );
    }
    return <CitationChip citation={citation} onCite={onCite} pending={pending} />;
  }
  return { a: CitationAnchor };
}
