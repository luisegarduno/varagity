"use client";

import type { GraphViewLegendEntry } from "@/lib/graph-view";
import { cn } from "@/lib/utils";

/**
 * The canvas legend: one pill per drawn entity type, in the same colour the
 * nodes wear (the map's legend bar, restyled for a WebGL canvas).
 *
 * The copy is deliberately explicit that these are **types, not
 * communities**: LightRAG builds no community tier at all (ADR-017), so
 * colour groups what the extraction actually labelled rather than a
 * clustering invented in the browser. Hovering or focusing a row spotlights
 * that type on the canvas.
 */
export function GraphLegend({
  entries,
  focused,
  onFocus,
}: {
  entries: GraphViewLegendEntry[];
  /** The entity type currently spotlighted, or `null`. */
  focused: string | null;
  onFocus: (entityType: string | null) => void;
}) {
  if (entries.length === 0) return null;
  return (
    <div className="pointer-events-auto absolute bottom-4 left-1/2 z-20 hidden max-w-[min(90%,44rem)] -translate-x-1/2 flex-wrap items-center justify-center gap-x-4 gap-y-1.5 rounded-full border border-border bg-card/80 px-5 py-2.5 backdrop-blur-md sm:flex">
      <span className="text-[10px] tracking-wider text-muted-foreground/70 uppercase">
        entity type
      </span>
      {entries.map((entry) => (
        <button
          key={entry.entityType}
          type="button"
          aria-pressed={focused === entry.entityType}
          onMouseEnter={() => onFocus(entry.entityType)}
          onMouseLeave={() => onFocus(null)}
          onFocus={() => onFocus(entry.entityType)}
          onBlur={() => onFocus(null)}
          onClick={() =>
            onFocus(focused === entry.entityType ? null : entry.entityType)
          }
          className={cn(
            "flex items-center gap-1.5 text-[11px] outline-none motion-safe:transition-colors",
            focused === entry.entityType
              ? "text-foreground"
              : "text-muted-foreground hover:text-foreground focus-visible:text-foreground",
          )}
        >
          <span
            aria-hidden
            className="size-2.5 shrink-0 rounded-full"
            style={{ backgroundColor: entry.color }}
          />
          {entry.entityType}
          <span className="tabular-nums opacity-60">{entry.count}</span>
        </button>
      ))}
    </div>
  );
}
