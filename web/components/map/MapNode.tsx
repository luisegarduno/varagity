/**
 * One node of the codebase map, drawn inside the SVG canvas
 * (spec_codebase_map.md §5.8, minus the dropped `cron` kind).
 *
 * Kinds are told apart by **shape + border + one accent-derived tint**, never
 * by color alone (a11y): a thick left bar for entries, a doubled border for
 * agents, a pill for models, a cylinder top for stores, a dotted border for
 * externals. A small lucide glyph and the design tokens do the rest, so the
 * map reads at both densities and in both themes without a bespoke palette.
 *
 * The interactive wrapper is a `<g role="button" tabIndex={0}>` with an
 * explicit `aria-label` — the button role hides the child `<text>` from the
 * a11y tree, and an unnamed focusable SVG element makes NVDA/JAWS repeat stale
 * names. `CodebaseMapView` owns selection; this component only renders and
 * reports clicks/Enter.
 */

import {
  BotIcon,
  CpuIcon,
  DatabaseIcon,
  GlobeIcon,
  LogInIcon,
  ServerIcon,
  WrenchIcon,
  type LucideIcon,
} from "lucide-react";

import type { MapNode as MapNodeModel, NodeKind } from "@/lib/codebase-map";
import type { NodeBox } from "@/lib/map-layout";
import { cn } from "@/lib/utils";

/** Legend-facing metadata for a kind: its glyph and human labels. */
export interface KindMeta {
  /** Human label for the legend and node `aria-label`. */
  label: string;
  /** The glyph drawn in the node's corner. */
  icon: LucideIcon;
  /** A one-phrase description of the shape treatment, for the legend. */
  treatment: string;
}

/** Kind → glyph + labels; the single source both nodes and the legend read. */
export const KIND_META: Record<NodeKind, KindMeta> = {
  entry: { label: "Entry point", icon: LogInIcon, treatment: "thick left bar" },
  agent: { label: "Agent", icon: BotIcon, treatment: "doubled border" },
  model: { label: "Model", icon: CpuIcon, treatment: "pill" },
  tool: { label: "Tool", icon: WrenchIcon, treatment: "compact card" },
  service: { label: "Service", icon: ServerIcon, treatment: "solid card" },
  store: { label: "Store", icon: DatabaseIcon, treatment: "cylinder top" },
  external: { label: "External", icon: GlobeIcon, treatment: "dotted border" },
};

const CORNER = 10;
/** How far a store's cap ellipse rises above the box (into the row gap). */
const STORE_CAP = 7;

/** SVG path for a store's cylinder body: straight sides, a bulging front. */
function cylinderBody(w: number, h: number): string {
  const cap = STORE_CAP;
  return `M 0 0 L 0 ${h - cap} A ${w / 2} ${cap} 0 0 0 ${w} ${h - cap} L ${w} 0 Z`;
}

/**
 * The kind-distinguishing shape only — no text — so both a placed node and a
 * tiny legend swatch can share exactly one rendering of "what an X looks like".
 *
 * @param props.kind - Which kind's shape to draw.
 * @param props.w - Box width.
 * @param props.h - Box height.
 * @param props.selected - Emphasize the border as the trace source.
 * @returns The SVG shape elements, positioned at the local origin.
 */
export function NodeShape({
  kind,
  w,
  h,
  selected = false,
}: {
  kind: NodeKind;
  w: number;
  h: number;
  selected?: boolean;
}) {
  const border = selected ? "stroke-primary" : "stroke-border";
  const weight = selected ? "stroke-2" : "stroke-[1.5]";

  switch (kind) {
    case "entry":
      return (
        <>
          <rect
            width={w}
            height={h}
            rx={CORNER}
            className={cn("fill-card", border, weight)}
          />
          <rect width={4} height={h} rx={2} className="fill-primary" />
        </>
      );
    case "agent":
      return (
        <>
          <rect
            width={w}
            height={h}
            rx={CORNER}
            className={cn("fill-card", border, weight)}
          />
          <rect
            x={1.5}
            y={1.5}
            width={w - 3}
            height={h - 3}
            rx={CORNER - 1.5}
            className="fill-primary/10"
          />
          <rect
            x={4}
            y={4}
            width={w - 8}
            height={h - 8}
            rx={CORNER - 4}
            className={cn("fill-none", border)}
            strokeWidth={1}
          />
        </>
      );
    case "model":
      return (
        <>
          <rect
            width={w}
            height={h}
            rx={h / 2}
            className={cn("fill-card", border, weight)}
          />
          <rect
            x={1.5}
            y={1.5}
            width={w - 3}
            height={h - 3}
            rx={h / 2}
            className="fill-primary/10"
          />
        </>
      );
    case "tool":
      return (
        <rect
          width={w}
          height={h}
          rx={CORNER}
          className={cn("fill-muted/40", border, weight)}
        />
      );
    case "service":
      return (
        <rect
          width={w}
          height={h}
          rx={CORNER}
          className={cn("fill-card", border, weight)}
        />
      );
    case "store":
      return (
        <>
          <path
            d={cylinderBody(w, h)}
            className={cn("fill-card", border, weight)}
          />
          <ellipse
            cx={w / 2}
            cy={0}
            rx={w / 2}
            ry={STORE_CAP}
            className={cn("fill-card", border, weight)}
          />
        </>
      );
    case "external":
      return (
        <rect
          width={w}
          height={h}
          rx={CORNER}
          strokeDasharray="2 3"
          className={cn(
            "fill-card",
            selected ? "stroke-primary" : "stroke-muted-foreground",
            weight,
          )}
        />
      );
  }
}

/** One placed, interactive node card. */
export function MapNode({
  node,
  box,
  selected,
  dimmed,
  clipId,
  onSelect,
}: {
  node: MapNodeModel;
  box: NodeBox;
  selected: boolean;
  dimmed: boolean;
  /** Shared clipPath id that keeps overflowing text inside the box. */
  clipId: string;
  onSelect: (id: string) => void;
}) {
  const Icon = KIND_META[node.kind].icon;
  const muted = node.kind === "tool";

  return (
    <g
      role="button"
      tabIndex={0}
      aria-label={`${node.label}, ${KIND_META[node.kind].label}`}
      aria-pressed={selected}
      transform={`translate(${box.x} ${box.y})`}
      onClick={() => onSelect(node.id)}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onSelect(node.id);
        }
      }}
      className={cn(
        "group cursor-pointer outline-none",
        "motion-safe:transition-opacity motion-safe:duration-300",
        dimmed ? "opacity-15" : "opacity-100",
      )}
    >
      <NodeShape kind={node.kind} w={box.w} h={box.h} selected={selected} />

      {/* Glyph in the top-right corner (inherits currentColor via `fill-*`). */}
      <g transform={`translate(${box.w - 24} 9)`} className="text-muted-foreground">
        <Icon width={14} height={14} aria-hidden />
      </g>

      {/* Text is clipped to the box so a long `sub` never spills into a gap. */}
      <g clipPath={`url(#${clipId})`}>
        <text
          x={14}
          y={24}
          className={cn(
            "text-[13px] font-medium",
            muted ? "fill-muted-foreground" : "fill-foreground",
          )}
        >
          {node.label}
        </text>
        {node.sub ? (
          <text x={14} y={41} className="fill-muted-foreground text-[11px]">
            {node.sub}
          </text>
        ) : null}
        {node.domain ? (
          <text
            x={14}
            y={box.h - 10}
            className="fill-muted-foreground font-mono text-[10px]"
          >
            {node.domain}
          </text>
        ) : null}
      </g>

      {/* Focus ring: reliable on SVG where `outline` is spotty. */}
      <rect
        width={box.w}
        height={box.h}
        rx={CORNER}
        pointerEvents="none"
        className="hidden fill-none stroke-ring stroke-2 group-focus-visible:block"
      />
    </g>
  );
}
