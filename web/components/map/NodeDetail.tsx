/**
 * The detail popover for a selected node — foglamp's treatment: a compact
 * card anchored beside the node with the kind row, the node's `detail`, and
 * its monospace `sourceRef`. The canvas spotlight (dim + trace) tells the
 * connection story; this is the reading surface.
 *
 * Positioned in *screen* space by the view (so it stays readable at any
 * zoom), and deliberately unfocusable chrome: Escape, a second click, or a
 * background click clears it.
 */

"use client";

import type { MapNode } from "@/lib/codebase-map";
import { cn } from "@/lib/utils";

import { Glyph, KIND_META } from "./MapNode";

/** Fixed popover width; the view uses it to clamp inside the canvas. */
export const DETAIL_WIDTH = 248;

/** The popover card, absolutely positioned over the canvas by the view. */
export function NodeDetail({
  node,
  left,
  top,
}: {
  node: MapNode;
  left: number;
  top: number;
}) {
  const meta = KIND_META[node.kind];
  return (
    <div
      data-map-popover
      style={{ left, top, width: DETAIL_WIDTH }}
      className="absolute z-30 rounded-xl border border-border bg-card p-4 text-card-foreground [box-shadow:var(--map-card-shadow)] dark:border-transparent"
    >
      <p
        className={cn(
          "flex items-center gap-1 text-[11px] font-medium tracking-wide",
          "text-muted-foreground",
        )}
      >
        <Glyph name={meta.glyph} className={cn("size-3", meta.tint)} />
        {meta.label}
      </p>
      <h2 className="mt-1 text-sm font-medium">{node.label}</h2>
      {(node.detail ?? node.sub) ? (
        <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
          {node.detail ?? node.sub}
        </p>
      ) : null}
      {node.sourceRef ? (
        <p className="mt-1.5 truncate font-mono text-[10px] text-muted-foreground/80">
          {node.sourceRef}
        </p>
      ) : null}
    </div>
  );
}
