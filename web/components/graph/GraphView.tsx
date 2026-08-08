"use client";

import { SigmaContainer } from "@react-sigma/core";
import { useQuery } from "@tanstack/react-query";
import Graph from "graphology";
import forceAtlas2 from "graphology-layout-forceatlas2";
import { NetworkIcon, PowerOffIcon, SearchIcon } from "lucide-react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useTheme } from "next-themes";
import { useCallback, useId, useMemo, useRef, useState } from "react";
import type Sigma from "sigma";
import { drawDiscNodeLabel } from "sigma/rendering";
import type { Settings } from "sigma/settings";
import type { EdgeDisplayData, NodeDisplayData } from "sigma/types";

import { GraphInspectPanel } from "@/components/graph/GraphInspectPanel";
import { GraphLegend } from "@/components/graph/GraphLegend";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ApiError } from "@/lib/api";
import {
  buildGraphView,
  filterEntities,
  neighborhood,
  parseMaxNodes,
  type GraphViewModel,
  type GraphViewNode,
} from "@/lib/graph-view";
import { graphExportQuery } from "@/lib/queries";
import { cn } from "@/lib/utils";

/** ForceAtlas2 passes. Enough to settle a few hundred nodes, cheap enough
 * to run synchronously before the first paint — which is also why there is
 * no layout animation for `prefers-reduced-motion` to suppress. */
const LAYOUT_ITERATIONS = 220;

/**
 * Camera ratio a search result lands at (smaller = closer). Deliberately
 * gentle: a hard zoom pushes the match's own neighbours off-screen, which
 * is the opposite of what "show me Bob" means. Never zooms *out* — an
 * already-closer camera keeps its ratio.
 */
const FOCUS_RATIO = 0.65;
const FOCUS_DURATION_MS = 400;

/** Node/edge alpha for the parts a hover or a legend focus pushes back. */
const DIMMED_NODE = "#c9ced8";
const DIMMED_EDGE = "#e2e5ea";
const DIMMED_NODE_DARK = "#3a4150";
const DIMMED_EDGE_DARK = "#2a3040";

/** What the reducers read on every frame, mutated outside React's render. */
interface Highlight {
  hovered: string | null;
  selected: string | null;
  /** The focused node's neighbourhood (itself included), or empty. */
  lit: Set<string>;
  /** The legend's spotlighted entity type, or `null`. */
  entityType: string | null;
}

/**
 * `/graph` — the whole extracted graph, drawn offline (spec_graphrag §4.4).
 *
 * Client-only by construction: sigma renders in WebGL, so the module is
 * behind `next/dynamic`'s `ssr: false` (see `GraphViewLoader`) — the repo's
 * first, because the `/map` ref-callback trick can defer *work* but not a
 * library's module evaluation.
 *
 * Three deliberate choices worth knowing:
 *
 * * **Colour is entity type, not community.** LightRAG builds no community
 *   layer (ADR-017), so the legend groups what the extraction actually
 *   labelled; the copy says so rather than implying clusters.
 * * **The layout is settled before the first paint.** ForceAtlas2 runs
 *   synchronously from a deterministic seed (`lib/graph-view.ts`), so the
 *   same graph draws the same picture on every reload, and there is no
 *   animation to suppress under reduced motion.
 * * **Highlighting never re-renders React.** Hover/selection live in a ref
 *   that sigma's reducers read, and a `refresh()` repaints — a settings
 *   change would rebuild the entire Sigma instance on every mouse move.
 */
export function GraphView() {
  const { resolvedTheme } = useTheme();
  const dark = resolvedTheme === "dark";
  const hintId = `${useId()}-graph-hint`;
  // `?max_nodes=` is the truncation-notice dev check (safe here without a
  // Suspense boundary: this component is `ssr: false`, so it never
  // prerenders — the only context where the hook would suspend).
  const maxNodes = parseMaxNodes(useSearchParams().get("max_nodes"));
  const { data, error, isPending } = useQuery(graphExportQuery(maxNodes));

  const [selected, setSelected] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [entityType, setEntityType] = useState<string | null>(null);
  const sigmaRef = useRef<Sigma | null>(null);
  const canvasRef = useRef<HTMLElement | null>(null);
  const highlightRef = useRef<Highlight>({
    hovered: null,
    selected: null,
    lit: new Set(),
    entityType: null,
  });

  const model: GraphViewModel = useMemo(
    () =>
      buildGraphView(data ?? { nodes: [], edges: [], truncated: false }),
    [data],
  );

  // The laid-out graphology instance. Rebuilt only when the export changes;
  // `SigmaContainer` recreates its renderer from it, which is exactly the
  // lifecycle we want (a new graph is a new picture).
  const graph = useMemo(() => {
    const instance = new Graph();
    for (const node of model.nodes) {
      instance.addNode(node.key, {
        label: node.label,
        x: node.x,
        y: node.y,
        size: node.size,
        color: node.color,
        entityType: node.entityType,
      });
    }
    for (const edge of model.edges) {
      instance.addEdgeWithKey(edge.key, edge.source, edge.target, {
        label: edge.label,
        size: edge.size,
        color: edge.color,
      });
    }
    if (instance.order > 0) {
      forceAtlas2.assign(instance, {
        iterations: LAYOUT_ITERATIONS,
        settings: forceAtlas2.inferSettings(instance),
      });
    }
    return instance;
  }, [model]);

  const settings = useMemo<Partial<Settings>>(
    () => ({
      // A hidden or still-collapsing parent must not throw on construction.
      allowInvalidContainer: true,
      renderEdgeLabels: false,
      labelColor: { color: dark ? "#e6e8ec" : "#1b1d21" },
      labelSize: 11,
      labelDensity: 0.6,
      labelGridCellSize: 70,
      // Labels sit to the *right* of their node, so the stock 30px gutter
      // clips the rightmost name on first fit.
      stagePadding: 64,
      // The stock hover renderer paints a hardcoded white card behind the
      // label — unreadable on the dark theme. Re-drawing the plain label is
      // the honest hover: the node is already lit by the reducers.
      defaultDrawNodeHover: drawDiscNodeLabel,
      zIndex: true,
      nodeReducer: (key: string, attributes) => {
        const { hovered, selected: pinned, lit, entityType: type } = highlightRef.current;
        const data = { ...attributes } as Partial<NodeDisplayData> & {
          entityType?: string;
        };
        if (type !== null && data.entityType !== type) {
          data.color = dark ? DIMMED_NODE_DARK : DIMMED_NODE;
          data.label = "";
          return data;
        }
        const focus = hovered ?? pinned;
        if (focus !== null && !lit.has(key)) {
          data.color = dark ? DIMMED_NODE_DARK : DIMMED_NODE;
          data.label = "";
        }
        if (key === pinned) {
          data.highlighted = true;
          data.zIndex = 1;
        }
        return data;
      },
      edgeReducer: (key: string, attributes) => {
        const { hovered, selected: pinned, entityType: type } = highlightRef.current;
        const data = { ...attributes } as Partial<EdgeDisplayData>;
        const focus = hovered ?? pinned;
        if (type !== null) {
          data.color = dark ? DIMMED_EDGE_DARK : DIMMED_EDGE;
          return data;
        }
        if (focus !== null) {
          const graph = sigmaRef.current?.getGraph();
          const touches =
            graph !== undefined &&
            (graph.source(key) === focus || graph.target(key) === focus);
          if (!touches) data.color = dark ? DIMMED_EDGE_DARK : DIMMED_EDGE;
        }
        return data;
      },
    }),
    [dark],
  );

  // Push new highlight state into the reducers and repaint. Closes over the
  // current edges, so it changes identity with the model — which is exactly
  // when `attach` should re-bind, the renderer having been rebuilt too.
  const edges = model.edges;
  const paint = useCallback(
    (next: Partial<Highlight>) => {
      const merged = { ...highlightRef.current, ...next };
      merged.lit = neighborhood(edges, merged.hovered ?? merged.selected);
      highlightRef.current = merged;
      sigmaRef.current?.refresh({ skipIndexation: true });
    },
    [edges],
  );

  const select = useCallback(
    (key: string | null) => {
      setSelected(key);
      paint({ selected: key });
    },
    [paint],
  );

  /** Centre the camera on one node — the search result's landing. */
  const focusNode = useCallback(
    (key: string) => {
      select(key);
      // The chosen result's button is about to unmount with the result list,
      // which would drop focus onto <body> and take Escape out of this
      // view's reach. Hand it to the canvas region instead.
      canvasRef.current?.focus();
      const sigma = sigmaRef.current;
      const position = sigma?.getNodeDisplayData(key);
      if (!sigma || !position) return;
      const reduced =
        typeof window !== "undefined" &&
        window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      const camera = sigma.getCamera();
      camera.animate(
        {
          x: position.x,
          y: position.y,
          ratio: Math.min(camera.getState().ratio, FOCUS_RATIO),
        },
        { duration: reduced ? 0 : FOCUS_DURATION_MS },
      );
    },
    [select],
  );

  /**
   * Attach to the renderer as it is created and detach as it dies. A
   * cleanup-returning ref callback is the house pattern for external
   * lifecycles (`/map`'s animation loop) — and here it is also the only
   * seam that sees the Sigma instance, which is born inside the container.
   */
  const attach = useCallback(
    (sigma: Sigma | null) => {
      sigmaRef.current = sigma;
      if (sigma === null) return;
      const onEnter = ({ node }: { node: string }) => paint({ hovered: node });
      const onLeave = () => paint({ hovered: null });
      const onClickNode = ({ node }: { node: string }) => {
        setSelected(node);
        paint({ selected: node });
      };
      const onClickStage = () => {
        setSelected(null);
        paint({ selected: null });
      };
      sigma.on("enterNode", onEnter);
      sigma.on("leaveNode", onLeave);
      sigma.on("clickNode", onClickNode);
      sigma.on("clickStage", onClickStage);
      // Re-apply whatever was lit before this renderer replaced the last one.
      paint({});
      return () => {
        sigma.off("enterNode", onEnter);
        sigma.off("leaveNode", onLeave);
        sigma.off("clickNode", onClickNode);
        sigma.off("clickStage", onClickStage);
        sigmaRef.current = null;
      };
    },
    [paint],
  );

  const matches: GraphViewNode[] = filterEntities(model.nodes, query);
  const empty = !isPending && error === null && model.nodes.length === 0;
  const disabled = error instanceof ApiError && error.code === "graph_disabled";

  return (
    <div
      className="flex h-full min-h-0 flex-col"
      onKeyDown={(event) => {
        if (event.key !== "Escape") return;
        if (query !== "") setQuery("");
        else if (selected !== null) select(null);
        else if (entityType !== null) {
          setEntityType(null);
          paint({ entityType: null });
        }
      }}
    >
      <header className="flex flex-wrap items-end justify-between gap-4 border-b border-border px-4 py-5 sm:px-6">
        <div className="min-w-0">
          <h1 className="font-heading text-2xl font-normal">Message graph</h1>
          <p className="mt-1 max-w-prose text-sm text-muted-foreground">
            Every entity the extraction found across the message archive, with
            the relations between them. Click a node to see what was said and
            when; colour is the entity type the extraction assigned.
          </p>
        </div>
        <label className="relative min-w-0 shrink-0">
          <span className="sr-only">Search entities</span>
          <SearchIcon
            aria-hidden
            className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground"
          />
          <input
            type="search"
            value={query}
            disabled={model.nodes.length === 0}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && matches.length > 0) {
                event.preventDefault();
                focusNode(matches[0].key);
                setQuery("");
              }
            }}
            placeholder="Search entities…"
            className="h-8 w-56 rounded-md border border-border bg-background pr-2.5 pl-8 text-sm outline-none focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/40 disabled:opacity-50"
          />
          {matches.length > 0 && (
            <ul
              aria-label="Search results"
              className="absolute top-full right-0 z-30 mt-1 w-72 overflow-hidden rounded-md border border-border bg-popover p-1 text-popover-foreground shadow-lg"
            >
              {matches.map((node) => (
                <li key={node.key}>
                  <button
                    type="button"
                    onClick={() => {
                      focusNode(node.key);
                      setQuery("");
                    }}
                    className="flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-left text-sm hover:bg-accent hover:text-accent-foreground focus-visible:bg-accent focus-visible:outline-none"
                  >
                    <span
                      aria-hidden
                      className="size-2 shrink-0 rounded-full"
                      style={{ backgroundColor: node.color }}
                    />
                    <span className="min-w-0 flex-1 truncate">{node.label}</span>
                    <span className="shrink-0 text-xs text-muted-foreground tabular-nums">
                      {node.degree}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </label>
      </header>

      {model.truncated && (
        <p
          role="status"
          className="border-b border-amber-500/30 bg-amber-500/10 px-4 py-2 text-xs sm:px-6"
        >
          Showing the {model.nodes.length.toLocaleString()} most connected
          entities — the graph holds more. Search still only sees what is
          drawn.
        </p>
      )}

      <div className="relative min-h-0 flex-1">
        {isPending ? (
          <div className="absolute inset-0 flex items-center justify-center p-6">
            <Skeleton className="size-full max-h-[36rem] max-w-4xl rounded-xl" />
          </div>
        ) : disabled ? (
          <EmptyState
            icon={<PowerOffIcon className="size-4 text-muted-foreground" />}
            title="The graph subsystem is off"
          >
            <code className="font-mono">GRAPH_ENABLED</code> is false, so the
            graph is not read rather than drawn half-empty. Turn it on in
            Settings — anything already extracted stays on disk.
          </EmptyState>
        ) : error !== null ? (
          <EmptyState
            icon={<NetworkIcon className="size-4 text-muted-foreground" />}
            title="Could not read the graph"
          >
            {error instanceof ApiError
              ? error.message
              : "API unreachable — is the stack up? (docker compose up -d)"}
          </EmptyState>
        ) : empty ? (
          <EmptyState
            icon={<NetworkIcon className="size-4 text-muted-foreground" />}
            title="Nothing extracted yet"
          >
            Upload a message archive and run a build — a full backfill runs
            for hours, and can be stopped and resumed.
            <span className="mt-3 block">
              <Button size="sm" variant="outline" render={<Link href="/corpus?tab=graph" />}>
                Open the graph corpus
              </Button>
            </span>
          </EmptyState>
        ) : (
          <>
            {/* Focusable so Tab reaches the canvas and Escape has somewhere
                to be pressed — the nodes themselves are WebGL pixels, so the
                search box above is the keyboard path *to* a node. */}
            <section
              ref={canvasRef}
              tabIndex={0}
              aria-label="Message graph canvas"
              aria-describedby={hintId}
              className="absolute inset-0 outline-none focus-visible:ring-2 focus-visible:ring-ring/50 focus-visible:ring-inset"
            >
              <SigmaContainer
                ref={attach}
                graph={graph}
                settings={settings}
                className="size-full"
              />
            </section>
            <p id={hintId} className="sr-only">
              Interactive graph of {model.nodes.length} entities and{" "}
              {model.edges.length} relations. Drag to pan, scroll to zoom.
              Search by name above and press Enter to centre a match; Escape
              clears the selection. Selecting an entity opens a panel with its
              relations and the transcript days it came from.
            </p>
            <GraphLegend
              entries={model.legend}
              focused={entityType}
              onFocus={(next) => {
                setEntityType(next);
                paint({ entityType: next });
              }}
            />
            {selected !== null && (
              <GraphInspectPanel name={selected} onClose={() => select(null)} />
            )}
          </>
        )}
      </div>
    </div>
  );
}

/** The centred "there is nothing to draw, and here is why" card. */
function EmptyState({
  icon,
  title,
  children,
}: {
  icon: React.ReactNode;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="absolute inset-0 flex items-center justify-center p-6">
      <section
        aria-label={title}
        className={cn(
          "flex max-w-md flex-col items-center gap-2 rounded-xl border border-dashed border-border p-8 text-center",
        )}
      >
        <span
          aria-hidden
          className="flex size-9 items-center justify-center rounded-full bg-muted"
        >
          {icon}
        </span>
        <p className="text-sm font-medium">{title}</p>
        <p className="text-xs leading-relaxed text-muted-foreground">
          {children}
        </p>
      </section>
    </div>
  );
}
