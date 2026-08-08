"use client";

import { ChevronRightIcon } from "lucide-react";
import { useState } from "react";

import { HighlightedText } from "@/components/provenance/ChunkCard";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsiblePanel,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import type {
  EvidenceChunk,
  GraphEvidence,
  GraphEvidenceEntity,
  GraphEvidenceRelation,
  GraphEvidenceTranscript,
} from "@/lib/evidence";
import { cn } from "@/lib/utils";

/**
 * Stable per-type tone. LightRAG has no community tier (ADR-017), so
 * `entity_type` is the grouping that honestly exists — a hash keeps the
 * same category the same colour across turns without a fixed vocabulary
 * the engine never promised.
 */
const TYPE_TONES = [
  "bg-sky-500/10 text-sky-700 dark:bg-sky-400/10 dark:text-sky-300",
  "bg-violet-500/10 text-violet-700 dark:bg-violet-400/10 dark:text-violet-300",
  "bg-emerald-500/10 text-emerald-700 dark:bg-emerald-400/10 dark:text-emerald-300",
  "bg-amber-500/10 text-amber-700 dark:bg-amber-400/10 dark:text-amber-300",
  "bg-rose-500/10 text-rose-700 dark:bg-rose-400/10 dark:text-rose-300",
  "bg-teal-500/10 text-teal-700 dark:bg-teal-400/10 dark:text-teal-300",
];

function typeTone(type: string | null): string {
  if (!type) return "bg-muted text-muted-foreground";
  let hash = 0;
  for (let index = 0; index < type.length; index += 1) {
    hash = (hash * 31 + type.charCodeAt(index)) >>> 0;
  }
  return TYPE_TONES[hash % TYPE_TONES.length];
}

/** The entities the retrieval surfaced, as type-toned chips. */
function EntityChips({ entities }: { entities: GraphEvidenceEntity[] }) {
  if (entities.length === 0) return null;
  return (
    <section aria-label="Entities" className="space-y-1.5">
      <h3 className="text-xs font-semibold text-muted-foreground">
        Entities ({entities.length})
      </h3>
      <ul className="flex flex-wrap gap-1">
        {entities.map((entity) => (
          <li key={`${entity.type ?? ""}:${entity.name}`}>
            <span
              title={
                [entity.type, entity.summary].filter(Boolean).join(" — ") ||
                undefined
              }
              className={cn(
                "inline-flex max-w-56 items-center rounded-md px-1.5 py-0.5 font-mono text-xs",
                typeTone(entity.type),
              )}
            >
              <span className="truncate">{entity.name}</span>
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}

/** The relations — the *facts* the answer was allowed to ground on. */
function RelationList({ relations }: { relations: GraphEvidenceRelation[] }) {
  if (relations.length === 0) return null;
  return (
    <section aria-label="Relations" className="space-y-1.5">
      <h3 className="text-xs font-semibold text-muted-foreground">
        Relations ({relations.length})
      </h3>
      <ul className="space-y-1">
        {relations.map((relation) => (
          <li
            key={`${relation.source}→${relation.target}:${relation.label ?? ""}`}
            className="rounded-md border border-border/60 px-2 py-1.5 text-xs"
          >
            <p className="flex flex-wrap items-center gap-1 font-mono text-[0.6875rem]">
              <span className="truncate font-medium">{relation.source}</span>
              <span aria-hidden className="text-muted-foreground">
                →
              </span>
              <span className="truncate font-medium">{relation.target}</span>
              {relation.label && (
                <Badge variant="outline" className="font-mono">
                  {relation.label}
                </Badge>
              )}
            </p>
            {relation.description && (
              <p className="mt-1 leading-relaxed text-muted-foreground">
                {relation.description}
              </p>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}

/**
 * One cited transcript day (spec_graphrag §4.3) — the Q2 "what was said
 * that day" surface, so the date is on the face of the card rather than
 * buried in a disclosure.
 *
 * Carries `data-chunk-key` and the shared card id because the citation
 * chips scroll to it through exactly the machinery chunk cards use; the
 * key is the day's `doc_key`.
 */
function TranscriptCard({
  chunk,
  transcript,
  query,
  className,
  style,
}: {
  /** The normalized evidence row (rank, key, excerpt, cite label). */
  chunk: EvidenceChunk;
  /** Its graph record — thread and span, split out for the card face. */
  transcript: GraphEvidenceTranscript | undefined;
  query: string | null;
  className?: string;
  style?: React.CSSProperties;
}) {
  const [expanded, setExpanded] = useState(false);
  const thread = transcript?.threadName ?? chunk.source ?? chunk.key;
  const span = transcript?.span ?? "";
  const messageCount = transcript?.messageCount ?? 0;
  return (
    <article
      id={`evidence-${chunk.key}`}
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
        {span && (
          <Badge variant="outline" className="font-mono">
            {span}
          </Badge>
        )}
      </header>

      <p className="flex flex-wrap items-center gap-x-1.5 gap-y-1 text-xs text-muted-foreground">
        <span
          className="max-w-full truncate font-medium text-foreground"
          title={chunk.source ?? undefined}
        >
          {thread}
        </span>
        {messageCount > 0 && (
          <span className="tabular-nums">
            {messageCount} message{messageCount === 1 ? "" : "s"}
          </span>
        )}
      </p>

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
          {expanded ? "Hide transcript" : "Show transcript"}
        </CollapsibleTrigger>
        <CollapsiblePanel>
          <p className="mt-1 rounded-md bg-muted/50 p-2 text-xs leading-relaxed whitespace-pre-wrap">
            <HighlightedText text={chunk.content} query={query} />
          </p>
        </CollapsiblePanel>
      </Collapsible>
    </article>
  );
}

/**
 * ★ A graph answer's evidence: what the engine knew (entities, relations)
 * and where it read it (the cited transcript days).
 *
 * Day-grain by construction — the engine attributes a conversation-day,
 * never a single message (ADR-017's priced regret), so the panel shows
 * days and the footer says so rather than inventing rows nothing can back.
 */
export function GraphEvidenceCards({
  graph,
  chunks,
  query,
  live,
}: {
  graph: GraphEvidence;
  /** The transcript days, already normalized into evidence rows. */
  chunks: EvidenceChunk[];
  query: string | null;
  /** Fresh evidence rises in card by card; reloaded answers render at rest. */
  live: boolean;
}) {
  return (
    <div className="space-y-3">
      <EntityChips entities={graph.entities} />
      <RelationList relations={graph.relations} />
      {chunks.length > 0 && (
        <section aria-label="Transcript days" className="space-y-1.5">
          <h3 className="text-xs font-semibold text-muted-foreground">
            Transcript days ({chunks.length})
          </h3>
          <div className="space-y-2">
            {chunks.map((chunk, index) => (
              <TranscriptCard
                key={chunk.key}
                chunk={chunk}
                transcript={graph.transcripts[index]}
                query={query}
                className={
                  live
                    ? "motion-safe:animate-in motion-safe:fade-in motion-safe:slide-in-from-bottom-2 motion-safe:fill-mode-backwards motion-safe:duration-300"
                    : undefined
                }
                style={
                  live
                    ? { animationDelay: `${Math.min(index, 8) * 40}ms` }
                    : undefined
                }
              />
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
