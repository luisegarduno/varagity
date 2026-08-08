"use client";

import { FilesIcon, NetworkIcon } from "lucide-react";
import type { ReactNode } from "react";

import {
  CORPUS_HINTS,
  CORPUS_LABELS,
  CORPUS_OPTIONS,
  type TargetCorpus,
} from "@/lib/corpus";
import { cn } from "@/lib/utils";

const ICONS: Record<TargetCorpus, ReactNode> = {
  rag: <FilesIcon aria-hidden className="size-3.5" />,
  graph: <NetworkIcon aria-hidden className="size-3.5" />,
};

const GRAPH_OFF_HINT =
  "The graph subsystem is off (GRAPH_ENABLED) — turn it on in Settings.";

/**
 * The composer's source selector (spec_graphrag §4.2): which corpus the
 * next question is answered from.
 *
 * A two-option segmented control rather than a menu — there are exactly
 * two corpora and the current one has to be readable at a glance, since it
 * silently decides what the answer can possibly know. It fills
 * `ChatRequest.corpus`, the same field a query router will fill later.
 *
 * With the graph subsystem off, the graph option is disabled and says why:
 * a graph-targeted turn would degrade to a document answer (ADR-017), and
 * a control that quietly does something else is worse than no control.
 */
export function SourceToggle({
  value,
  onChange,
  graphEnabled,
  disabled = false,
}: {
  value: TargetCorpus;
  onChange: (corpus: TargetCorpus) => void;
  /** `GRAPH_ENABLED` — the kill switch, read from the settings catalog. */
  graphEnabled: boolean;
  disabled?: boolean;
}) {
  return (
    <div
      role="group"
      aria-label="Answer from"
      className="inline-flex items-center gap-0.5 rounded-lg bg-muted p-0.5"
    >
      {CORPUS_OPTIONS.map((corpus) => {
        const off = corpus === "graph" && !graphEnabled;
        const selected = value === corpus;
        return (
          <button
            key={corpus}
            type="button"
            aria-pressed={selected}
            aria-label={`Answer from ${CORPUS_LABELS[corpus]}`}
            title={off ? GRAPH_OFF_HINT : CORPUS_HINTS[corpus]}
            disabled={disabled || off}
            onClick={() => onChange(corpus)}
            className={cn(
              "inline-flex h-6 cursor-pointer items-center gap-1.5 rounded-md px-2 text-xs font-medium whitespace-nowrap transition-colors",
              "focus-visible:ring-2 focus-visible:ring-ring/50 focus-visible:outline-none",
              // Not `pointer-events-none`: the disabled option's `title`
              // (the "why is this off" hint) only shows on hover.
              "disabled:cursor-not-allowed disabled:opacity-50",
              selected
                ? "bg-background text-foreground shadow-xs"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {ICONS[corpus]}
            {CORPUS_LABELS[corpus]}
          </button>
        );
      })}
    </div>
  );
}
