/**
 * The map legend: the project summary header plus a kind → shape key
 * (spec_codebase_map.md §5.5 / §5.8). Each swatch reuses {@link NodeShape},
 * so "what an X looks like" is drawn in exactly one place.
 */

import type { CodebaseMap, NodeKind } from "@/lib/codebase-map";

import { KIND_META, NodeShape } from "./MapNode";

const KIND_ORDER: NodeKind[] = [
  "entry",
  "agent",
  "model",
  "tool",
  "service",
  "store",
  "external",
];

/** A tiny, non-interactive swatch of a kind's shape for the key. */
function Swatch({ kind }: { kind: NodeKind }) {
  return (
    <svg
      viewBox="-8 -8 76 56"
      width={34}
      height={24}
      aria-hidden
      className="shrink-0"
    >
      <NodeShape kind={kind} w={60} h={40} />
    </svg>
  );
}

/** The legend card, absolutely positioned over the canvas by the view. */
export function MapLegend({ project }: { project: CodebaseMap["project"] }) {
  return (
    <div className="pointer-events-auto max-w-72 rounded-xl border border-border bg-card/90 p-3.5 text-card-foreground shadow-sm backdrop-blur-sm">
      <h1 className="font-heading text-lg leading-tight font-normal">
        {project.name} · codebase map
      </h1>
      <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
        {project.summary}
      </p>
      <ul className="mt-3 grid grid-cols-2 gap-x-3 gap-y-1.5">
        {KIND_ORDER.map((kind) => (
          <li key={kind} className="flex items-center gap-2">
            <Swatch kind={kind} />
            <span className="min-w-0 text-[11px]">
              <span className="block truncate font-medium">
                {KIND_META[kind].label}
              </span>
              <span className="block truncate text-muted-foreground">
                {KIND_META[kind].treatment}
              </span>
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
