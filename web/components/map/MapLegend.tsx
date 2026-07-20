/**
 * The map's legend: the frosted pill bar along the bottom edge, one entry per
 * rendered kind (foglamp's treatment). Hovering or focusing an entry
 * spotlights that kind on the canvas — the view passes the highlight state
 * back down as per-node dimming.
 */

"use client";

import type { NodeKind } from "@/lib/codebase-map";
import { cn } from "@/lib/utils";

import { Glyph, KIND_META } from "./MapNode";

/** The kinds that render as cards, in legend order (models fold into chips). */
const LEGEND_KINDS: NodeKind[] = ["entry", "agent", "service", "store", "external"];

/** The legend bar, absolutely positioned over the canvas by the view. */
export function MapLegend({
  onKindFocus,
}: {
  /** Called with a kind to spotlight, or `null` to clear. */
  onKindFocus: (kind: NodeKind | null) => void;
}) {
  return (
    <div className="pointer-events-auto absolute bottom-4 left-1/2 z-20 hidden -translate-x-1/2 items-center gap-4 rounded-full bg-card/70 px-5 py-2.5 backdrop-blur-md [box-shadow:var(--map-card-shadow)] sm:flex">
      {LEGEND_KINDS.map((kind) => {
        const meta = KIND_META[kind];
        return (
          <button
            key={kind}
            type="button"
            aria-label={`Highlight ${meta.plural.toLowerCase()}`}
            onMouseEnter={() => onKindFocus(kind)}
            onMouseLeave={() => onKindFocus(null)}
            onFocus={() => onKindFocus(kind)}
            onBlur={() => onKindFocus(null)}
            className={cn(
              "flex cursor-default items-center gap-1.5 text-[10px] font-medium tracking-wider uppercase",
              "text-muted-foreground/70 outline-none hover:text-foreground focus-visible:text-foreground",
              "motion-safe:transition-colors",
            )}
          >
            <Glyph name={meta.glyph} className={cn("size-3.5", meta.tint)} />
            {meta.plural}
          </button>
        );
      })}
    </div>
  );
}
