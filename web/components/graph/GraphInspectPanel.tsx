"use client";

import { useQuery } from "@tanstack/react-query";
import { CalendarDaysIcon, XIcon } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ApiError } from "@/lib/api";
import { entityTypeColor, normalizeEntityType } from "@/lib/graph-view";
import { graphEntityQuery } from "@/lib/queries";

/**
 * The click-through panel (spec_graphrag §4.4): what the engine merged
 * about one entity, the relations around it, and — the part that makes the
 * graph answerable — **the transcript days it came from**.
 *
 * Those days are the drill-down the spec's Q2 asks for ("what was the
 * conversation that day?"), and they are honest about their grain: the
 * engine attributes a day span, not a message, so the card names the thread
 * and the span and says how many messages the manifest accounts for.
 */
export function GraphInspectPanel({
  name,
  onClose,
}: {
  name: string;
  onClose: () => void;
}) {
  const { data, error, isPending } = useQuery(graphEntityQuery(name));
  const entityType = normalizeEntityType(data?.entity.entity_type);

  return (
    <aside
      aria-label={`Entity ${name}`}
      className="pointer-events-auto absolute top-4 right-4 bottom-4 z-20 flex w-80 max-w-[calc(100%-2rem)] flex-col overflow-hidden rounded-xl border border-border bg-card/95 text-card-foreground shadow-lg backdrop-blur-sm"
    >
      <header className="flex items-start gap-2 border-b border-border px-4 py-3">
        <span
          aria-hidden
          className="mt-1.5 size-2.5 shrink-0 rounded-full"
          style={{ backgroundColor: entityTypeColor(entityType) }}
        />
        <div className="min-w-0 flex-1">
          <h2 className="font-heading text-base leading-snug font-normal break-words">
            {name}
          </h2>
          <p className="mt-0.5 text-[11px] text-muted-foreground">
            {entityType}
            {data ? ` · ${data.relations.length} relations` : ""}
          </p>
        </div>
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label="Close entity panel"
          onClick={onClose}
        >
          <XIcon className="size-4" />
        </Button>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3 text-sm">
        {isPending ? (
          <div className="flex flex-col gap-2">
            <Skeleton className="h-3 w-full" />
            <Skeleton className="h-3 w-4/5" />
            <Skeleton className="h-3 w-2/3" />
          </div>
        ) : error ? (
          <p role="alert" className="text-xs text-destructive">
            {error instanceof ApiError && error.code === "entity_not_found"
              ? "The graph no longer holds this entity — it may have been rebuilt since this picture was drawn."
              : error instanceof ApiError
                ? error.message
                : "Could not read this entity — is the stack up?"}
          </p>
        ) : (
          <>
            {data.entity.description && (
              <p className="leading-relaxed text-muted-foreground">
                {data.entity.description}
              </p>
            )}

            <h3 className="mt-4 mb-1.5 text-xs font-semibold">Relations</h3>
            {data.relations.length === 0 ? (
              <p className="text-xs text-muted-foreground">
                No relations in this slice.
              </p>
            ) : (
              <ul className="flex flex-col gap-1.5">
                {data.relations.map((relation) => {
                  const other =
                    relation.source === data.entity.id
                      ? relation.target
                      : relation.source;
                  return (
                    <li
                      key={relation.id}
                      className="rounded-md border border-border/60 px-2.5 py-1.5 text-xs"
                    >
                      <span className="flex flex-wrap items-center gap-1.5">
                        {relation.label && (
                          <Badge variant="outline" className="font-normal">
                            {relation.label}
                          </Badge>
                        )}
                        <span className="font-medium break-words">{other}</span>
                      </span>
                      {relation.description && (
                        <p className="mt-1 leading-relaxed text-muted-foreground">
                          {relation.description}
                        </p>
                      )}
                    </li>
                  );
                })}
              </ul>
            )}

            <h3 className="mt-4 mb-1.5 text-xs font-semibold">Source days</h3>
            {data.sources.length === 0 ? (
              <p className="text-xs text-muted-foreground">
                The engine recorded no source documents for this entity.
              </p>
            ) : (
              <ul className="flex flex-col gap-1.5">
                {data.sources.map((source) => (
                  <li
                    key={source.doc_key}
                    className="flex items-start gap-2 rounded-md border border-border/60 px-2.5 py-1.5 text-xs"
                  >
                    <CalendarDaysIcon
                      aria-hidden
                      className="mt-0.5 size-3.5 shrink-0 text-muted-foreground"
                    />
                    <span className="min-w-0">
                      <span className="block font-medium break-words">
                        {source.thread_name}
                      </span>
                      <span className="block font-mono text-[11px] text-muted-foreground tabular-nums">
                        {source.span}
                        {source.message_count > 0
                          ? ` · ${source.message_count} messages`
                          : ""}
                      </span>
                    </span>
                  </li>
                ))}
              </ul>
            )}
            <p className="mt-3 text-[11px] leading-relaxed text-muted-foreground">
              Evidence is day-grain: the graph attributes an entity to the
              conversation day it was extracted from, not to a single message.
            </p>
          </>
        )}
      </div>
    </aside>
  );
}
