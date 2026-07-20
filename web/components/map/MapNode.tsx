/**
 * One node card of the codebase map — the foglamp-style treatment: a squircle
 * card with a kind-colored icon badge (or the integration's vendored favicon),
 * a title + sub line, and the model chips the layout folded into it. Agent
 * cards carry the animated `map-beam` border (globals.css).
 *
 * Rendered as a real, absolutely-positioned `<button>` inside the pan/zoom
 * world, so keyboard focus, Enter/Space activation, and accessible naming are
 * native — no SVG `role="button"` workarounds. `CodebaseMapView` owns
 * selection; this component only renders and reports clicks.
 */

"use client";

import { useState } from "react";

import type { MapNode as MapNodeModel, NodeKind } from "@/lib/codebase-map";
import type { ModelChip, NodeBox } from "@/lib/map-layout";
import { cn } from "@/lib/utils";

/**
 * Filled glyph paths from the MIT-licensed Tabler icon set, inlined because
 * the house lucide set has no filled variants and the foglamp look leans on
 * solid marks. Keys are the Tabler icon names.
 */
export const GLYPH_PATHS = {
  "bolt-filled": [
    "M13 2l.018 .001l.016 .001l.083 .005l.011 .002h.011l.038 .009l.052 .008l.016 .006l.011 .001l.029 .011l.052 .014l.019 .009l.015 .004l.028 .014l.04 .017l.021 .012l.022 .01l.023 .015l.031 .017l.034 .024l.018 .011l.013 .012l.024 .017l.038 .034l.022 .017l.008 .01l.014 .012l.036 .041l.026 .027l.006 .009c.12 .147 .196 .322 .218 .513l.001 .012l.002 .041l.004 .064v6h5a1 1 0 0 1 .868 1.497l-.06 .091l-8 11c-.568 .783 -1.808 .38 -1.808 -.588v-6h-5a1 1 0 0 1 -.868 -1.497l.06 -.091l8 -11l.01 -.013l.018 -.024l.033 -.038l.018 -.022l.009 -.008l.013 -.014l.04 -.036l.028 -.026l.008 -.006a1 1 0 0 1 .402 -.199l.011 -.001l.027 -.005l.074 -.013l.011 -.001l.041 -.002z",
  ],
  "ghost-filled": [
    "M12 3a8 8 0 0 1 7.996 7.75l.004 .25l-.001 6.954l.01 .103a2.78 2.78 0 0 1 -1.468 2.618l-.163 .08c-1.053 .475 -2.283 .248 -3.129 -.593l-.137 -.146a.65 .65 0 0 0 -1.024 0a2.65 2.65 0 0 1 -4.176 0a.65 .65 0 0 0 -.512 -.25c-.2 0 -.389 .092 -.55 .296a2.78 2.78 0 0 1 -4.859 -2.005l.008 -.091l.001 -6.966l.004 -.25a8 8 0 0 1 7.996 -7.75zm2.82 10.429a1 1 0 0 0 -1.391 -.25a2.5 2.5 0 0 1 -2.858 0a1 1 0 0 0 -1.142 1.642a4.5 4.5 0 0 0 5.142 0a1 1 0 0 0 .25 -1.392zm-4.81 -4.429l-.127 .007a1 1 0 0 0 .117 1.993l.127 -.007a1 1 0 0 0 -.117 -1.993zm4 0l-.127 .007a1 1 0 0 0 .117 1.993l.127 -.007a1 1 0 0 0 -.117 -1.993z",
  ],
  "hexagon-filled": [
    "M10.425 1.414l-6.775 3.996a3.21 3.21 0 0 0 -1.65 2.807v7.285a3.226 3.226 0 0 0 1.678 2.826l6.695 4.237c1.034 .57 2.22 .57 3.2 .032l6.804 -4.302c.98 -.537 1.623 -1.618 1.623 -2.793v-7.284l-.005 -.204a3.223 3.223 0 0 0 -1.284 -2.39l-.107 -.075l-.007 -.007a1.074 1.074 0 0 0 -.181 -.133l-6.776 -3.995a3.33 3.33 0 0 0 -3.216 0z",
  ],
  "database-filled": [
    "M3 15.731c1.968 1.507 5.234 2.269 9 2.269c3.76 0 7.025 -.76 9 -2.252v2.252c0 2.425 -3.895 3.936 -8.693 3.998l-.307 .002c-4.938 0 -9 -1.523 -9 -4z",
    "M3 9.731c1.968 1.507 5.234 2.269 9 2.269c3.76 0 7.025 -.76 9 -2.252v2.252c0 2.477 -4.062 4 -9 4c-4.798 0 -8.77 -1.438 -8.979 -3.795l-.016 -.101l-.005 -.104z",
    "M12 2c1.041 0 2.044 .068 2.977 .198l.469 .071q .84 .14 1.586 .348l.44 .131l.075 .024a11 11 0 0 1 .805 .3l.199 .086q .535 .242 .967 .53q .165 .11 .313 .225a3.8 3.8 0 0 1 .669 .668l.091 .128q .07 .105 .129 .211l.07 .139q .163 .35 .2 .73l.01 .211c0 2.477 -4.062 4 -9 4c-4.798 0 -8.77 -1.438 -8.979 -3.795a1 1 0 0 1 -.021 -.205l.005 -.104l.016 -.1c.205 -2.306 4.01 -3.733 8.667 -3.794z",
  ],
  "world-filled": [
    "M21.165 16a10 10 0 0 1 -8.434 5.973a1 1 0 0 0 .617 -.444a18 18 0 0 0 2.28 -5.528z",
    "M8.372 16a18 18 0 0 0 2.28 5.53a1 1 0 0 0 .616 .443a10 10 0 0 1 -8.433 -5.973z",
    "M13.57 16a16 16 0 0 1 -1.57 3.884a16 16 0 0 1 -1.57 -3.884",
    "M8.034 10a18 18 0 0 0 0 4h-5.832a10 10 0 0 1 -.002 -4z",
    "M13.952 10a16 16 0 0 1 0 4h-3.904a16 16 0 0 1 0 -4z",
    "M21.8 10a10.05 10.05 0 0 1 -.002 4h-5.832c.149 -1.329 .149 -2.67 0 -4z",
    "M11.267 2.027a1 1 0 0 0 -.615 .444a18 18 0 0 0 -2.28 5.529h-5.54a10.01 10.01 0 0 1 8.334 -5.967z",
    "M12 4.116a16 16 0 0 1 1.57 3.885h-3.14c.34 -1.317 .85 -2.6 1.53 -3.817z",
    "M12.733 2.026a10.01 10.01 0 0 1 8.435 5.974h-5.54a18 18 0 0 0 -2.28 -5.53a1 1 0 0 0 -.517 -.414z",
  ],
  "ai-agent": [
    "M11 14a1 1 0 1 0 2 0a1 1 0 1 0 -2 0",
    "M6 14a1 1 0 1 0 2 0a1 1 0 1 0 -2 0",
    "M16 14a1 1 0 1 0 2 0a1 1 0 1 0 -2 0",
    "M11 5a1 1 0 1 0 2 0a1 1 0 1 0 -2 0",
    "M13.5 9.5a1 1 0 1 0 2 0a1 1 0 1 0 -2 0",
    "M8.5 9.5a1 1 0 1 0 2 0a1 1 0 1 0 -2 0",
    "M13.5 18.5a1 1 0 1 0 2 0a1 1 0 1 0 -2 0",
    "M8.5 18.5a1 1 0 1 0 2 0a1 1 0 1 0 -2 0",
    "M3.5 18.5a1 1 0 1 0 2 0a1 1 0 1 0 -2 0",
    "M18.5 18.5a1 1 0 1 0 2 0a1 1 0 1 0 -2 0",
  ],
  "sitemap-filled": [
    "M2 16.667a2.667 2.667 0 0 1 2.667 -2.667h2.666a2.667 2.667 0 0 1 2.667 2.667v2.666a2.667 2.667 0 0 1 -2.667 2.667h-2.666a2.667 2.667 0 0 1 -2.667 -2.667z",
    "M14 16.667a2.667 2.667 0 0 1 2.667 -2.667h2.666a2.667 2.667 0 0 1 2.667 2.667v2.666a2.667 2.667 0 0 1 -2.667 2.667h-2.666a2.667 2.667 0 0 1 -2.667 -2.667z",
    "M8 4.667a2.667 2.667 0 0 1 2.667 -2.667h2.666a2.667 2.667 0 0 1 2.667 2.667v2.666a2.667 2.667 0 0 1 -2.667 2.667h-2.666a2.667 2.667 0 0 1 -2.667 -2.667z",
    "M12 8a1 1 0 0 0 -1 1v2h-3c-1.645 0 -3 1.355 -3 3v1a1 1 0 0 0 1 1a1 1 0 0 0 1 -1v-1c0 -.564 .436 -1 1 -1h8c.564 0 1 .436 1 1v1a1 1 0 0 0 1 1a1 1 0 0 0 1 -1v-1c0 -1.645 -1.355 -3 -3 -3h-3v-2a1 1 0 0 0 -1 -1z",
  ],
} as const;

/** One inlined filled glyph, sized by the caller via `className`. */
export function Glyph({
  name,
  className,
}: {
  name: keyof typeof GLYPH_PATHS;
  className?: string;
}) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="currentColor"
      aria-hidden
      className={cn("size-4", className)}
    >
      {GLYPH_PATHS[name].map((d) => (
        <path key={d.slice(0, 24)} d={d} />
      ))}
    </svg>
  );
}

/** Legend/detail-facing metadata for a kind: glyph, label, and palette. */
export interface KindMeta {
  /** Human label for the legend, node `aria-label`, and detail popover. */
  label: string;
  /** Plural label the legend bar shows. */
  plural: string;
  /** The badge glyph. */
  glyph: keyof typeof GLYPH_PATHS;
  /** Icon badge background + foreground classes. */
  badge: string;
  /** Standalone icon tint (legend bar, detail popover). */
  tint: string;
  /** Hover/selection ring color class. */
  ring: string;
  /** Raw color (Tailwind 500) — edge beam comets and hit-ring flashes. */
  hex: string;
}

/** Kind → visual identity; the single source every map surface reads. */
export const KIND_META: Record<NodeKind, KindMeta> = {
  entry: {
    label: "Trigger",
    plural: "Triggers",
    glyph: "bolt-filled",
    badge: "bg-muted text-foreground",
    tint: "text-amber-600 dark:text-amber-400",
    ring: "border-slate-500",
    hex: "#64748b",
  },
  agent: {
    label: "Agent",
    plural: "Agents",
    glyph: "ghost-filled",
    badge: "bg-orange-500/10 text-orange-600 dark:text-orange-400",
    tint: "text-orange-600 dark:text-orange-400",
    ring: "border-orange-500",
    hex: "#f97316",
  },
  model: {
    label: "Model",
    plural: "Models",
    glyph: "ai-agent",
    badge: "bg-blue-500/10 text-blue-600 dark:text-blue-400",
    tint: "text-blue-600 dark:text-blue-400",
    ring: "border-blue-500",
    hex: "#3b82f6",
  },
  tool: {
    label: "Tool",
    plural: "Tools",
    glyph: "hexagon-filled",
    badge: "bg-violet-500/10 text-violet-600 dark:text-violet-400",
    tint: "text-violet-600 dark:text-violet-400",
    ring: "border-violet-500",
    hex: "#8b5cf6",
  },
  service: {
    label: "Service",
    plural: "Services",
    glyph: "hexagon-filled",
    badge: "bg-pink-500/10 text-pink-600 dark:text-pink-400",
    tint: "text-pink-600 dark:text-pink-400",
    ring: "border-pink-500",
    hex: "#ec4899",
  },
  store: {
    label: "Store",
    plural: "Stores",
    glyph: "database-filled",
    badge: "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400",
    tint: "text-emerald-600 dark:text-emerald-400",
    ring: "border-emerald-500",
    hex: "#10b981",
  },
  external: {
    label: "External",
    plural: "External",
    glyph: "world-filled",
    badge: "bg-sky-500/10 text-sky-600 dark:text-sky-400",
    tint: "text-sky-600 dark:text-sky-400",
    ring: "border-sky-500",
    hex: "#0ea5e9",
  },
};

/**
 * Resolve a favicon domain to its vendored asset (`public/map-icons/`) —
 * the map never fetches icons from the network (house privacy rule; the
 * assets were vendored once at build-authoring time).
 */
export function iconSrc(domain: string): string {
  return `/map-icons/${domain}.png`;
}

/** A vendored favicon that falls back to the kind glyph if the asset 404s. */
function Favicon({
  domain,
  kind,
  className,
}: {
  domain: string;
  kind: NodeKind;
  className?: string;
}) {
  const [failed, setFailed] = useState(false);
  if (failed) return <Glyph name={KIND_META[kind].glyph} className={className} />;
  return (
    // Static vendored asset inside a zoomable canvas; next/image adds nothing.
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={iconSrc(domain)}
      alt=""
      onError={() => setFailed(true)}
      className={cn("rounded-[5px] object-contain", className)}
    />
  );
}

/** One model chip row inside a card. */
function Chip({ chip }: { chip: ModelChip }) {
  const domain = chip.domain;
  return (
    <span className="flex max-w-full items-center gap-1.5">
      {domain ? (
        <Favicon domain={domain} kind="model" className="size-3" />
      ) : (
        <Glyph name="ai-agent" className="size-3 text-muted-foreground" />
      )}
      <span className="truncate text-xs font-medium">{chip.label}</span>
    </span>
  );
}

/** One placed, interactive node card. */
export function MapNodeCard({
  node,
  box,
  chips,
  selected,
  dimmed,
  enterDelay,
  hitRingRef,
  onSelect,
}: {
  node: MapNodeModel;
  box: NodeBox;
  chips: ModelChip[] | undefined;
  selected: boolean;
  dimmed: boolean;
  /** Entrance stagger (s) — things appear along the flow direction. */
  enterDelay: number;
  /** Registers the beam-hit ring overlay, flashed when a beam arrives. */
  hitRingRef: (el: HTMLSpanElement | null) => void;
  onSelect: (id: string) => void;
}) {
  const meta = KIND_META[node.kind];
  return (
    <button
      type="button"
      data-map-node={node.id}
      aria-label={`${node.label}, ${meta.label}`}
      aria-pressed={selected}
      onClick={() => onSelect(node.id)}
      style={{
        left: box.x,
        top: box.y,
        width: box.w,
        height: box.h,
        animationDelay: `${enterDelay}s`,
        animationFillMode: "backwards",
      }}
      className={cn(
        "group map-node-enter absolute cursor-pointer rounded-[22px] text-left outline-none",
        "focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
        "motion-safe:transition-[opacity,filter] motion-safe:duration-300",
        dimmed ? "opacity-25 blur-[1px]" : "opacity-100",
      )}
    >
      <span
        className={cn(
          "flex h-full flex-col overflow-hidden rounded-[22px] bg-card text-card-foreground",
          "border border-border [box-shadow:var(--map-card-shadow)] dark:border-transparent",
          node.kind === "agent" && "map-beam",
        )}
      >
        <span className="flex h-14 flex-none items-center gap-2.5 px-3.5">
          <span
            className={cn(
              "flex size-7 flex-none items-center justify-center rounded-[10px]",
              meta.badge,
            )}
          >
            {node.domain ? (
              <Favicon domain={node.domain} kind={node.kind} className="size-[18px]" />
            ) : (
              <Glyph name={meta.glyph} />
            )}
          </span>
          <span className="flex min-w-0 flex-col">
            <span className="truncate text-sm leading-snug font-medium">
              {node.label}
            </span>
            {node.sub ? (
              <span className="truncate text-xs leading-snug text-muted-foreground">
                {node.sub}
              </span>
            ) : null}
          </span>
        </span>
        {chips && chips.length > 0 ? (
          <span className="mx-4 flex flex-1 flex-col items-start gap-2 border-t border-muted pt-2.5 pb-2.5">
            {chips.map((chip) => (
              <Chip key={chip.label} chip={chip} />
            ))}
          </span>
        ) : null}
      </span>
      {/* Kind-colored ring, revealed on hover and while selected. */}
      <span
        aria-hidden
        className={cn(
          "pointer-events-none absolute inset-0 rounded-[22px] border",
          "motion-safe:transition-opacity motion-safe:duration-300",
          meta.ring,
          selected ? "opacity-100" : "opacity-0 group-hover:opacity-70",
        )}
      />
      {/* Beam-hit ring — flashed imperatively in the arriving beam's color. */}
      <span
        ref={hitRingRef}
        aria-hidden
        className="pointer-events-none absolute inset-0 rounded-[22px] border opacity-0 transition-[opacity,border-color] duration-500"
        style={{ borderColor: "transparent" }}
      />
    </button>
  );
}
