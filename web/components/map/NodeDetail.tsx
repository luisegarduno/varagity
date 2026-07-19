/**
 * The click-through detail panel for a selected node (spec_codebase_map.md
 * §5.5 / §5.7): its `detail`, a monospace `sourceRef`, and the node's in/out
 * edges rendered as sentences ("calls bge-reranker-v2-m3 — cross-encode,
 * keep 5"). The trace itself lives on the canvas; this is the reading surface.
 */

import { XIcon } from "lucide-react";

import type { CodebaseMap, EdgeKind, MapNode } from "@/lib/codebase-map";
import { Button } from "@/components/ui/button";

import { KIND_META } from "./MapNode";

/** How each edge kind reads as a verb in a sentence. */
const EDGE_VERB: Record<EdgeKind, string> = {
  calls: "calls",
  reads: "reads",
  writes: "writes to",
  triggers: "triggers",
};

function verb(kind: EdgeKind | undefined): string {
  return kind ? EDGE_VERB[kind] : "connects to";
}

/** One edge rendered as a sentence, with the other endpoint emphasized. */
function EdgeSentence({
  lead,
  target,
  label,
}: {
  lead: string;
  target: string;
  label?: string;
}) {
  return (
    <li className="text-xs leading-relaxed text-muted-foreground">
      {lead} <span className="font-medium text-foreground">{target}</span>
      {label ? <span> — {label}</span> : null}
    </li>
  );
}

/** The detail card, absolutely positioned over the canvas by the view. */
export function NodeDetail({
  node,
  map,
  onClose,
}: {
  node: MapNode;
  map: CodebaseMap;
  onClose: () => void;
}) {
  const labelOf = (id: string): string =>
    map.graph.nodes.find((candidate) => candidate.id === id)?.label ?? id;

  const outgoing = map.graph.edges.filter((edge) => edge.from === node.id);
  const incoming = map.graph.edges.filter((edge) => edge.to === node.id);
  const meta = [node.sub, node.group].filter(Boolean).join(" · ");

  return (
    <div className="pointer-events-auto flex max-h-full w-80 max-w-full flex-col overflow-hidden rounded-xl border border-border bg-card/95 text-card-foreground shadow-lg backdrop-blur-sm">
      <div className="flex items-start justify-between gap-2 border-b border-border p-3.5">
        <div className="min-w-0">
          <h2 className="font-heading text-lg leading-tight font-normal">
            {node.label}
          </h2>
          <p className="mt-0.5 text-xs text-muted-foreground">
            {KIND_META[node.kind].label}
            {meta ? ` · ${meta}` : ""}
          </p>
        </div>
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label="Close details"
          onClick={onClose}
        >
          <XIcon />
        </Button>
      </div>

      <div className="flex min-h-0 flex-col gap-3 overflow-y-auto p-3.5">
        {node.detail ? (
          <p className="text-sm leading-relaxed">{node.detail}</p>
        ) : null}

        {node.domain ? (
          <p className="font-mono text-xs text-muted-foreground">
            {node.domain}
          </p>
        ) : null}

        {node.sourceRef ? (
          <div>
            <p className="text-[10px] font-medium tracking-wide text-muted-foreground uppercase">
              source
            </p>
            <p className="mt-0.5 font-mono text-xs break-all">
              {node.sourceRef}
            </p>
          </div>
        ) : null}

        {outgoing.length > 0 ? (
          <div>
            <p className="text-[10px] font-medium tracking-wide text-muted-foreground uppercase">
              downstream
            </p>
            <ul className="mt-1 flex flex-col gap-1">
              {outgoing.map((edge) => (
                <EdgeSentence
                  key={`${edge.from}-${edge.to}`}
                  lead={verb(edge.kind)}
                  target={labelOf(edge.to)}
                  label={edge.label}
                />
              ))}
            </ul>
          </div>
        ) : null}

        {incoming.length > 0 ? (
          <div>
            <p className="text-[10px] font-medium tracking-wide text-muted-foreground uppercase">
              upstream
            </p>
            <ul className="mt-1 flex flex-col gap-1">
              {incoming.map((edge) => (
                <EdgeSentence
                  key={`${edge.from}-${edge.to}`}
                  lead={`${labelOf(edge.from)} ${verb(edge.kind)}`}
                  target="this"
                  label={edge.label}
                />
              ))}
            </ul>
          </div>
        ) : null}
      </div>
    </div>
  );
}
