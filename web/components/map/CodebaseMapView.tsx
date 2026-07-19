"use client";

import { FrameIcon, ZoomInIcon, ZoomOutIcon } from "lucide-react";
import { useCallback, useId, useMemo, useRef, useState } from "react";

import { CODEBASE_MAP } from "@/lib/codebase-map.data";
import type { MapEdge } from "@/lib/codebase-map";
import { layout } from "@/lib/map-layout";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

import { KIND_META, MapNode } from "./MapNode";
import { MapLegend } from "./MapLegend";
import { NodeDetail } from "./NodeDetail";

/** Zoom clamps and step (spec_codebase_map.md §5.7: 0.4×–2.5×). */
const MIN_SCALE = 0.4;
const MAX_SCALE = 2.5;
const ZOOM_STEP = 1.2;
/** Trackpad/wheel zoom sensitivity per wheel delta unit. */
const WHEEL_BASE = 1.0015;

/** The pan/zoom of the inner drawing group, in the SVG's user coordinates. */
interface Transform {
  x: number;
  y: number;
  scale: number;
}

const IDENTITY: Transform = { x: 0, y: 0, scale: 1 };

function clamp(value: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, value));
}

/**
 * Apply a zoom `factor` about a fixed user-space point, so the content under
 * that point stays put (cursor for wheel, view center for keys/buttons).
 */
function zoomAbout(prev: Transform, ux: number, uy: number, factor: number): Transform {
  const scale = clamp(prev.scale * factor, MIN_SCALE, MAX_SCALE);
  const applied = scale / prev.scale;
  return {
    scale,
    x: ux - (ux - prev.x) * applied,
    y: uy - (uy - prev.y) * applied,
  };
}

/** Map a client (screen) point into the SVG's user coordinate system. */
function clientToUser(
  svg: SVGSVGElement,
  clientX: number,
  clientY: number,
): { x: number; y: number } {
  const ctm = svg.getScreenCTM();
  if (!ctm) return { x: clientX, y: clientY };
  const point = new DOMPoint(clientX, clientY).matrixTransform(ctm.inverse());
  return { x: point.x, y: point.y };
}

/**
 * The codebase map: a deterministic layered SVG of how Varagity fits together,
 * with pan, zoom-to-cursor, and click-to-trace of downstream flows
 * (spec_codebase_map.md §5.5–§5.8). Reachable from `/map`; the developer-mode
 * gating of its entry points arrives in a later phase.
 *
 * No `useEffect` (the house rule): selection is `useState`, the layout and
 * adjacency are `useMemo` over a static import, pan/zoom live in event
 * handlers, and the one native listener — the non-passive `wheel`, needed
 * because React's synthetic `onWheel` is forcibly passive
 * (facebook/react#19654) — is attached in a cleanup-returning ref callback.
 */
export function CodebaseMapView() {
  const map = CODEBASE_MAP;
  // The input never changes at runtime, so the whole layout is computed once.
  const { positions, edgePaths, groupBands, bounds } = useMemo(() => layout(map), [map]);

  const [transform, setTransform] = useState<Transform>(IDENTITY);
  const [selected, setSelected] = useState<string | null>(null);

  const svgRef = useRef<SVGSVGElement | null>(null);
  const panRef = useRef<{ pointerId: number; x: number; y: number } | null>(null);

  const reactId = useId();
  const hintId = `${reactId}-hint`;
  const clipId = `${reactId}-clip`;
  const arrowId = `${reactId}-arrow`;
  const arrowActiveId = `${reactId}-arrow-active`;

  const nodesById = useMemo(
    () => new Map(map.graph.nodes.map((node) => [node.id, node])),
    [map],
  );

  // from → its outgoing edges: drives both the trace closure and the SR mirror.
  const edgesByFrom = useMemo(() => {
    const out = new Map<string, MapEdge[]>();
    for (const edge of map.graph.edges) {
      const list = out.get(edge.from);
      if (list) list.push(edge);
      else out.set(edge.from, [edge]);
    }
    return out;
  }, [map]);

  // The transitive-downstream closure of the selected node (it, plus every
  // node any edge chain reaches). `null` when nothing is selected.
  const reachable = useMemo<ReadonlySet<string> | null>(() => {
    if (selected === null) return null;
    const seen = new Set<string>([selected]);
    const stack = [selected];
    while (stack.length > 0) {
      const id = stack.pop() as string;
      for (const edge of edgesByFrom.get(id) ?? []) {
        if (!seen.has(edge.to)) {
          seen.add(edge.to);
          stack.push(edge.to);
        }
      }
    }
    return seen;
  }, [selected, edgesByFrom]);

  // One clip rect for every node (uniform box sizes) keeps long text inside.
  const clip = useMemo(() => {
    let w = 0;
    let h = 0;
    for (const box of positions.values()) {
      w = Math.max(w, box.w);
      h = Math.max(h, box.h);
    }
    return { w, h };
  }, [positions]);

  const toggleSelect = useCallback((id: string) => {
    setSelected((prev) => (prev === id ? null : id));
  }, []);

  const zoomAboutCenter = useCallback(
    (factor: number) => {
      setTransform((prev) => zoomAbout(prev, bounds.w / 2, bounds.h / 2, factor));
    },
    [bounds.w, bounds.h],
  );

  // Attach the non-passive wheel listener (and stash the element for pan's
  // coordinate math) in a cleanup-returning ref callback. Idempotent so Strict
  // Mode's attach → cleanup → attach cycle nets a single live listener.
  const attachCanvas = useCallback((el: SVGSVGElement | null) => {
    svgRef.current = el;
    if (el === null) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const point = clientToUser(el, event.clientX, event.clientY);
      setTransform((prev) =>
        zoomAbout(prev, point.x, point.y, Math.pow(WHEEL_BASE, -event.deltaY)),
      );
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => {
      el.removeEventListener("wheel", onWheel);
      svgRef.current = null;
    };
  }, []);

  function startPan(event: React.PointerEvent<SVGRectElement>) {
    const svg = svgRef.current;
    if (!svg) return;
    const point = clientToUser(svg, event.clientX, event.clientY);
    panRef.current = { pointerId: event.pointerId, x: point.x, y: point.y };
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function movePan(event: React.PointerEvent<SVGRectElement>) {
    const pan = panRef.current;
    const svg = svgRef.current;
    if (!pan || !svg || pan.pointerId !== event.pointerId) return;
    const point = clientToUser(svg, event.clientX, event.clientY);
    const dx = point.x - pan.x;
    const dy = point.y - pan.y;
    pan.x = point.x;
    pan.y = point.y;
    setTransform((prev) => ({ ...prev, x: prev.x + dx, y: prev.y + dy }));
  }

  function endPan(event: React.PointerEvent<SVGRectElement>) {
    if (panRef.current?.pointerId === event.pointerId) {
      event.currentTarget.releasePointerCapture(event.pointerId);
      panRef.current = null;
    }
  }

  // Zoom keys and Escape bubble up from the focused node or button.
  function onKeyDown(event: React.KeyboardEvent<HTMLDivElement>) {
    switch (event.key) {
      case "Escape":
        if (selected !== null) setSelected(null);
        break;
      case "+":
      case "=":
        zoomAboutCenter(ZOOM_STEP);
        event.preventDefault();
        break;
      case "-":
      case "_":
        zoomAboutCenter(1 / ZOOM_STEP);
        event.preventDefault();
        break;
      case "0":
        setTransform(IDENTITY);
        event.preventDefault();
        break;
      default:
        break;
    }
  }

  const selectedNode = selected === null ? null : (nodesById.get(selected) ?? null);
  const labelOf = (id: string): string => nodesById.get(id)?.label ?? id;

  return (
    <div
      className="relative min-h-0 flex-1 overflow-hidden bg-background"
      onKeyDown={onKeyDown}
    >
      <svg
        ref={attachCanvas}
        role="application"
        aria-label="Codebase map"
        aria-describedby={hintId}
        viewBox={`0 0 ${bounds.w} ${bounds.h}`}
        preserveAspectRatio="xMidYMid meet"
        className="h-full w-full touch-none select-none"
      >
        <defs>
          <clipPath id={clipId}>
            <rect width={clip.w} height={clip.h} />
          </clipPath>
          <marker
            id={arrowId}
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="7"
            markerHeight="7"
            orient="auto-start-reverse"
          >
            <path d="M 0 1 L 9 5 L 0 9 z" className="fill-foreground/40" />
          </marker>
          <marker
            id={arrowActiveId}
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="7"
            markerHeight="7"
            orient="auto-start-reverse"
          >
            <path d="M 0 1 L 9 5 L 0 9 z" className="fill-primary" />
          </marker>
        </defs>

        {/* Full-viewBox background: catches pan gestures at any zoom. */}
        <rect
          x={0}
          y={0}
          width={bounds.w}
          height={bounds.h}
          fill="transparent"
          className="cursor-grab active:cursor-grabbing"
          onPointerDown={startPan}
          onPointerMove={movePan}
          onPointerUp={endPan}
          onPointerCancel={endPan}
        />

        <g
          transform={`translate(${transform.x} ${transform.y}) scale(${transform.scale})`}
        >
          {/* Group bands, drawn behind their members. */}
          {groupBands.map((band) => (
            <g key={band.group}>
              <rect
                x={band.x}
                y={band.y}
                width={band.w}
                height={band.h}
                rx={16}
                strokeDasharray="4 5"
                className="fill-muted/25 stroke-border"
                strokeWidth={1}
              />
              <text
                x={band.x + 16}
                y={band.y + 18}
                className="fill-muted-foreground text-[11px] font-medium tracking-wide uppercase"
              >
                {band.group}
              </text>
            </g>
          ))}

          {/* Edges. On trace, downstream edges brighten and reveal their kind. */}
          {edgePaths.map(({ edge, d }, index) => {
            const active = reachable !== null && reachable.has(edge.from);
            const dim = reachable !== null && !active;
            const from = positions.get(edge.from);
            const to = positions.get(edge.to);
            const mx = from && to ? (from.x + from.w + to.x) / 2 : 0;
            const my =
              from && to ? (from.y + from.h / 2 + to.y + to.h / 2) / 2 : 0;
            const chipWidth = (edge.kind?.length ?? 1) * 6 + 12;
            return (
              <g
                key={index}
                className={cn(
                  "motion-safe:transition-opacity motion-safe:duration-300",
                  dim ? "opacity-15" : "opacity-100",
                )}
              >
                <path
                  d={d}
                  fill="none"
                  markerEnd={`url(#${active ? arrowActiveId : arrowId})`}
                  className={cn(
                    active
                      ? "stroke-primary stroke-[1.75]"
                      : "stroke-muted-foreground/45 stroke-[1.25]",
                  )}
                />
                {active && edge.kind ? (
                  <g
                    transform={`translate(${mx} ${my})`}
                    className="pointer-events-none motion-safe:animate-in motion-safe:fade-in motion-safe:duration-300"
                  >
                    <rect
                      x={-chipWidth / 2}
                      y={-8}
                      width={chipWidth}
                      height={16}
                      rx={5}
                      className="fill-popover stroke-border"
                      strokeWidth={1}
                    />
                    <text
                      textAnchor="middle"
                      y={4}
                      className="fill-muted-foreground font-mono text-[9px]"
                    >
                      {edge.kind}
                    </text>
                  </g>
                ) : null}
              </g>
            );
          })}

          {/* Nodes, in declaration (reading) order so Tab follows the story. */}
          {map.graph.nodes.map((node) => {
            const box = positions.get(node.id);
            if (!box) return null;
            return (
              <MapNode
                key={node.id}
                node={node}
                box={box}
                clipId={clipId}
                selected={selected === node.id}
                dimmed={reachable !== null && !reachable.has(node.id)}
                onSelect={toggleSelect}
              />
            );
          })}
        </g>
      </svg>

      {/* Usage hint the SVG's aria-describedby points at. */}
      <p id={hintId} className="sr-only">
        Interactive diagram. Tab to reach a node; Enter or Space traces its
        downstream flow; Escape clears; plus, minus, and zero zoom. A text
        equivalent of every node and its connections follows.
      </p>

      {/* Browse-mode text equivalent (not aria-hidden; no focusables). */}
      <ul className="sr-only">
        {map.graph.nodes.map((node) => {
          const outs = edgesByFrom.get(node.id) ?? [];
          return (
            <li key={node.id}>
              {node.label} ({KIND_META[node.kind].label})
              {node.sub ? `: ${node.sub}` : ""}.
              {outs.length > 0
                ? ` Connects to ${outs
                    .map(
                      (edge) =>
                        `${labelOf(edge.to)}${edge.label ? ` (${edge.label})` : ""}`,
                    )
                    .join(", ")}.`
                : ""}
            </li>
          );
        })}
      </ul>

      <div className="pointer-events-none absolute inset-0 flex flex-col justify-between p-4">
        <div className="flex items-start justify-between gap-4">
          <MapLegend project={map.project} />
          {selectedNode ? (
            <NodeDetail
              node={selectedNode}
              map={map}
              onClose={() => setSelected(null)}
            />
          ) : null}
        </div>

        <div className="pointer-events-auto flex w-fit items-center gap-1 rounded-lg border border-border bg-card/90 p-1 shadow-sm backdrop-blur-sm">
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label="Zoom in"
            onClick={() => zoomAboutCenter(ZOOM_STEP)}
          >
            <ZoomInIcon />
          </Button>
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label="Zoom out"
            onClick={() => zoomAboutCenter(1 / ZOOM_STEP)}
          >
            <ZoomOutIcon />
          </Button>
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label="Reset view"
            onClick={() => setTransform(IDENTITY)}
          >
            <FrameIcon />
          </Button>
        </div>
      </div>
    </div>
  );
}
