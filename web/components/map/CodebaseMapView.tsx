"use client";

import { FrameIcon, ZoomInIcon, ZoomOutIcon } from "lucide-react";
import { useCallback, useId, useMemo, useRef, useState } from "react";

import { CODEBASE_MAP } from "@/lib/codebase-map.data";
import { precomputedLayout } from "@/lib/codebase-map.layout";
import type { MapEdge, NodeKind } from "@/lib/codebase-map";
import { arrowHead, edgePath, labelAnchor, polylineLength } from "@/lib/map-layout";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

import { KIND_META, MapNodeCard } from "./MapNode";
import { MapLegend } from "./MapLegend";
import { MapSidePanel } from "./MapSidePanel";
import { NodeDetail } from "./NodeDetail";

// Mixed from the OPAQUE background (never from --border, which carries alpha
// in dark mode): translucent strokes stack where routes share a channel,
// reading as doubled lines. Opaque same-color strokes overlap invisibly.
const EDGE_STROKE =
  "color-mix(in oklab, var(--background) 70%, var(--muted-foreground) 30%)";

const MIN_SCALE = 0.2;
const MAX_SCALE = 3;
const ZOOM_STEP = 1.2;
/** Trackpad pinch sensitivity (ctrlKey+wheel). */
const PINCH_BASE = 0.012;
/** Space the fit keeps clear of the floating side panel on wide viewports. */
const FIT_PAD_LEFT = 320;
const FIT_PAD_RIGHT = 48;
const FIT_PAD_Y = 56;
/** Pointer travel below this is a click, not a pan. */
const CLICK_SLOP = 4;

// The traveling beams: a glowing comet occasionally runs an edge and "hits"
// the target (flashing its ring). Sparse by design — one run, long rest.
const BEAM_SPEED = 0.25; // px per ms — an unhurried glide
const BEAM_TRAVEL_MIN_MS = 1600;
const BEAM_TRAVEL_MAX_MS = 5000;
const BEAM_REST_MS = 30000; // + up to 25s jitter between runs

interface Transform {
  x: number;
  y: number;
  k: number;
}

const clamp = (v: number, lo: number, hi: number): number =>
  Math.min(hi, Math.max(lo, v));

/**
 * A discrete traveling beam: a comet (bright leading edge, fading tail) that
 * glides along the edge path via a WAAPI offset-path animation —
 * compositor-friendly, so even the full map stays smooth. The animation loop
 * lives in a cleanup-returning ref callback (the house `useEffect`-free
 * pattern) and never starts under reduced motion.
 */
function BeamPulse({
  d,
  length,
  color,
  dimmed,
  targetId,
  onArrive,
}: {
  d: string;
  length: number;
  color: string;
  dimmed: boolean;
  /** The endpoint whose ring flashes when the comet lands. */
  targetId: string;
  /** Stable across renders — captured once by the animation loop. */
  onArrive: (targetId: string, color: string) => void;
}) {
  const attach = useCallback(
    (el: HTMLDivElement | null) => {
      if (!el || typeof el.animate !== "function") return;
      if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
      let stopped = false;
      let anim: Animation | null = null;
      let timer: ReturnType<typeof setTimeout>;
      const travelMs = clamp(length / BEAM_SPEED, BEAM_TRAVEL_MIN_MS, BEAM_TRAVEL_MAX_MS);
      const run = (): void => {
        if (stopped) return;
        anim?.cancel(); // release the previous run's forward-fill
        el.style.opacity = "1";
        // fill: "forwards" holds the beam at the target while the 0.25s
        // opacity fade plays — without it the beam snaps back to the path
        // start and flashes there mid-fade.
        anim = el.animate(
          [{ offsetDistance: "0%" }, { offsetDistance: "100%" }],
          { duration: travelMs, easing: "ease-in-out", fill: "forwards" },
        );
        anim.onfinish = () => {
          el.style.opacity = "0";
          if (stopped) return;
          onArrive(targetId, color);
          timer = setTimeout(run, BEAM_REST_MS + Math.random() * 25000);
        };
      };
      timer = setTimeout(run, 1200 + Math.random() * BEAM_REST_MS);
      return () => {
        stopped = true;
        clearTimeout(timer);
        anim?.cancel();
      };
    },
    [length, targetId, color, onArrive],
  );
  return (
    <div className={cn("transition-opacity duration-300", dimmed && "opacity-15")}>
      {/* A comet, not a dot: a pill rotated along the path (offsetRotate auto
          keeps its x-axis on the travel direction). */}
      <div
        ref={attach}
        className="pointer-events-none absolute top-0 left-0 h-[3px] w-8 rounded-full"
        style={{
          offsetPath: `path("${d}")`,
          offsetRotate: "auto",
          background: `linear-gradient(to right, transparent, ${color})`,
          boxShadow: `0 0 10px ${color}66`,
          opacity: 0,
          transition: "opacity 0.25s ease",
        }}
      />
    </div>
  );
}

/**
 * The codebase map: foglamp's flow-map treatment on the house tokens — ELK
 * lays the cards and containers out (see lib/map-layout.ts), edges render as
 * soft orthogonal runs with chevrons and traveling beam comets, and clicking
 * a card spotlights its downstream flow with an anchored detail popover.
 *
 * No `useEffect` (the house rule): the layout is a synchronous import of the
 * drift-guarded snapshot, pan/zoom write a ref's transform straight to the
 * DOM in commit-phase ref callbacks (running it through React state
 * re-rendered the whole graph every pointermove), and the two native
 * listeners (non-passive `wheel`, the beams' WAAPI loops) attach in
 * cleanup-returning ref callbacks.
 */
export function CodebaseMapView() {
  const map = CODEBASE_MAP;
  // The checked-in, drift-guarded ELK snapshot — synchronous, so the map is
  // interactive the moment it hydrates (no elkjs in the browser).
  const layout = useMemo(() => precomputedLayout(), []);
  const { nodes, foldedEdges, positions, groupBands, edges, chips, bounds } = layout;

  const [selected, setSelected] = useState<string | null>(null);
  const [kindFocus, setKindFocus] = useState<NodeKind | null>(null);

  const reactId = useId();
  const hintId = `${reactId}-hint`;

  // ── Pan/zoom, imperatively ────────────────────────────────────────────────
  const graphRef = useRef<HTMLDivElement | null>(null);
  const popoverRef = useRef<HTMLDivElement | null>(null);
  const viewportSize = useRef<{ w: number; h: number } | null>(null);
  const tRef = useRef<Transform>({ x: 24, y: 24, k: 0.55 });
  const tracedPosRef = useRef<{ x: number; y: number } | null>(null);
  const fitted = useRef(false);
  const drag = useRef<{
    pointerId: number;
    px: number;
    py: number;
    tx: number;
    ty: number;
    moved: boolean;
  } | null>(null);

  // `will-change: transform` is held only while a gesture is in flight — kept
  // permanently, the browser caches one raster and zoom goes blurry.
  const idleTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const applyTransform = useCallback(() => {
    const t = tRef.current;
    const g = graphRef.current;
    if (g) {
      g.style.willChange = "transform";
      g.style.transform = `translate(${t.x}px, ${t.y}px) scale(${t.k})`;
      g.style.opacity = "1";
      if (idleTimer.current) clearTimeout(idleTimer.current);
      idleTimer.current = setTimeout(() => {
        g.style.willChange = "auto";
      }, 150);
    }
    const pop = popoverRef.current;
    const pos = tracedPosRef.current;
    if (pop && pos) {
      pop.style.left = `${t.x + pos.x * t.k}px`;
      pop.style.top = `${t.y + pos.y * t.k}px`;
    }
  }, []);

  const zoomAboutCenter = useCallback(
    (factor: number) => {
      const size = viewportSize.current;
      const cx = size ? size.w / 2 : 0;
      const cy = size ? size.h / 2 : 0;
      const prev = tRef.current;
      const k = clamp(prev.k * factor, MIN_SCALE, MAX_SCALE);
      const ratio = k / prev.k;
      tRef.current = {
        k,
        x: cx - (cx - prev.x) * ratio,
        y: cy - (cy - prev.y) * ratio,
      };
      applyTransform();
    },
    [applyTransform],
  );

  // Fit the graph into the visible area (clear of the floating panel) once.
  // Capped at 1 — upscaling blows the cards past native size. A graph too
  // deep to fit readably instead fits its height and opens on the start of
  // the flow, panning right through the story.
  const fitToViewport = useCallback(
    (el: HTMLDivElement) => {
      const wide = el.clientWidth >= 900;
      const padL = wide ? FIT_PAD_LEFT : 24;
      const padR = wide ? FIT_PAD_RIGHT : 24;
      const availW = Math.max(200, el.clientWidth - padL - padR);
      const availH = Math.max(200, el.clientHeight - FIT_PAD_Y * 2);
      const kFit = Math.min(availW / bounds.w, availH / bounds.h);
      if (kFit >= 0.45) {
        const k = clamp(kFit, 0.3, 1);
        tRef.current = {
          x: padL + (availW - bounds.w * k) / 2,
          y: FIT_PAD_Y + (availH - bounds.h * k) / 2,
          k,
        };
      } else {
        const k = clamp((availH / bounds.h) * 0.9, 0.5, 0.8);
        tRef.current = {
          x: padL + 16,
          y: FIT_PAD_Y + (availH - bounds.h * k) / 2,
          k,
        };
      }
      applyTransform();
    },
    [bounds.w, bounds.h, applyTransform],
  );

  // Measure, fit once, and attach the non-passive wheel listener — all in a
  // cleanup-returning ref callback (React's onWheel is forcibly passive).
  // Idempotent under Strict Mode's attach → cleanup → attach cycle.
  const attachViewport = useCallback(
    (el: HTMLDivElement | null) => {
      if (el === null) return;
      viewportSize.current = { w: el.clientWidth, h: el.clientHeight };
      if (!fitted.current) {
        fitted.current = true;
        fitToViewport(el);
      }
      const onWheel = (event: WheelEvent) => {
        event.preventDefault();
        const prev = tRef.current;
        if (event.ctrlKey) {
          // Trackpad pinch arrives as ctrlKey+wheel — the gesture that zooms.
          const rect = el.getBoundingClientRect();
          const cx = event.clientX - rect.left;
          const cy = event.clientY - rect.top;
          const factor = Math.exp(-event.deltaY * PINCH_BASE);
          const k = clamp(prev.k * factor, MIN_SCALE, MAX_SCALE);
          const ratio = k / prev.k;
          tRef.current = {
            k,
            x: cx - (cx - prev.x) * ratio,
            y: cy - (cy - prev.y) * ratio,
          };
        } else {
          // Two-finger scroll (either axis) pans the canvas.
          tRef.current = {
            ...prev,
            x: prev.x - event.deltaX,
            y: prev.y - event.deltaY,
          };
        }
        applyTransform();
      };
      el.addEventListener("wheel", onWheel, { passive: false });
      return () => {
        el.removeEventListener("wheel", onWheel);
      };
    },
    [fitToViewport, applyTransform],
  );

  function isCanvasTarget(target: EventTarget | null): boolean {
    return !(
      target instanceof Element && target.closest("button, [data-map-popover], a")
    );
  }

  function startPan(event: React.PointerEvent<HTMLDivElement>) {
    if (!isCanvasTarget(event.target)) return;
    drag.current = {
      pointerId: event.pointerId,
      px: event.clientX,
      py: event.clientY,
      tx: tRef.current.x,
      ty: tRef.current.y,
      moved: false,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
  }
  function movePan(event: React.PointerEvent<HTMLDivElement>) {
    const d = drag.current;
    if (!d || d.pointerId !== event.pointerId) return;
    if (Math.hypot(event.clientX - d.px, event.clientY - d.py) > CLICK_SLOP) {
      d.moved = true;
    }
    if (!d.moved) return;
    tRef.current = {
      ...tRef.current,
      x: d.tx + (event.clientX - d.px),
      y: d.ty + (event.clientY - d.py),
    };
    applyTransform();
  }
  function endPan(event: React.PointerEvent<HTMLDivElement>) {
    const d = drag.current;
    if (d?.pointerId !== event.pointerId) return;
    event.currentTarget.releasePointerCapture(event.pointerId);
    const wasDrag = d.moved;
    drag.current = null;
    // A stationary click on the background clears the spotlight.
    if (!wasDrag && isCanvasTarget(event.target) && selected !== null) {
      setSelected(null);
    }
  }

  // Zoom keys and Escape bubble up from the focused card or button.
  function onKeyDown(event: React.KeyboardEvent<HTMLElement>) {
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
      case "0": {
        fitted.current = false;
        const viewport = graphRef.current?.parentElement;
        if (viewport instanceof HTMLDivElement) {
          fitted.current = true;
          fitToViewport(viewport);
        }
        event.preventDefault();
        break;
      }
      default:
        break;
    }
  }

  const nodesById = useMemo(() => new Map(nodes.map((n) => [n.id, n])), [nodes]);

  // Trace: clicking a card lights its full downstream path (BFS along the
  // folded edges) and opens the detail popover.
  const trace = useMemo(() => {
    if (selected === null) return null;
    const inTrace = new Set<string>([selected]);
    const edgeSet = new Set<number>();
    const queue = [selected];
    while (queue.length > 0) {
      const cur = queue.shift() as string;
      foldedEdges.forEach((e, i) => {
        if (e.from !== cur) return;
        edgeSet.add(i);
        if (!inTrace.has(e.to)) {
          inTrace.add(e.to);
          queue.push(e.to);
        }
      });
    }
    return { nodes: inTrace, edges: edgeSet };
  }, [selected, foldedEdges]);

  const nodeActive = (id: string, kind: NodeKind): boolean =>
    (kindFocus === null || kind === kindFocus) &&
    (trace === null || trace.nodes.has(id));
  const edgeActive = (e: { orig: number[] }): boolean =>
    (kindFocus === null ||
      e.orig.some((i) => {
        const o = foldedEdges[i];
        return (
          nodesById.get(o.from)?.kind === kindFocus ||
          nodesById.get(o.to)?.kind === kindFocus
        );
      })) &&
    (trace === null || e.orig.some((i) => trace.edges.has(i)));

  // Beam-hit rings: one overlay per node/group, flashed imperatively when a
  // beam arrives (refs, not state — a flash shouldn't re-render the graph).
  const hitRings = useRef(new Map<string, HTMLElement>());
  const hitTimers = useRef(new Map<string, ReturnType<typeof setTimeout>>());
  const registerHitRing = (id: string) => (el: HTMLElement | null) => {
    if (el) hitRings.current.set(id, el);
    else hitRings.current.delete(id);
  };
  const flashTarget = useCallback((id: string, color: string) => {
    const el = hitRings.current.get(id);
    if (!el) return;
    el.style.borderColor = color;
    el.style.opacity = "0.55"; // soft glow, not a hard outline
    const prev = hitTimers.current.get(id);
    if (prev) clearTimeout(prev);
    hitTimers.current.set(
      id,
      setTimeout(() => {
        el.style.opacity = "0";
      }, 600),
    );
  }, []);

  // Entrance choreography: things appear along the flow direction.
  const delayAt = (x: number, y = 0): number =>
    0.15 + ((x + y) / Math.max(1, bounds.w + bounds.h)) * 0.9;
  const xOf = useMemo(() => {
    const m = new Map<string, number>();
    for (const [id, box] of positions) m.set(id, box.x);
    for (const g of groupBands) m.set(g.id, g.x);
    return m;
  }, [positions, groupBands]);

  const selectedNode = selected === null ? null : (nodesById.get(selected) ?? null);
  const selectedBox = selected === null ? undefined : positions.get(selected);

  // The graph's transform is never rendered by React — it is applied in this
  // commit-phase ref callback and re-applied imperatively on every gesture,
  // so a pan/zoom frame never re-renders the graph.
  const attachGraph = useCallback(
    (el: HTMLDivElement | null) => {
      graphRef.current = el;
      if (el) applyTransform();
    },
    [applyTransform],
  );

  // Same deal for the popover: anchored under its card at mount (the wrapper
  // is keyed by node id, so a new selection re-runs this), then kept glued by
  // applyTransform() while panning/zooming.
  const attachPopover = useCallback(
    (el: HTMLDivElement | null) => {
      popoverRef.current = el;
      if (el === null) {
        tracedPosRef.current = null;
        return;
      }
      const box = el.dataset.anchor ? JSON.parse(el.dataset.anchor) : null;
      if (box) {
        tracedPosRef.current = { x: box.x, y: box.y + box.h + 10 };
        const t = tRef.current;
        el.style.left = `${t.x + tracedPosRef.current.x * t.k}px`;
        el.style.top = `${t.y + tracedPosRef.current.y * t.k}px`;
      }
    },
    [],
  );

  const labelOf = (id: string): string => nodesById.get(id)?.label ?? id;

  return (
    <section
      aria-label="Codebase map"
      aria-describedby={hintId}
      onKeyDown={onKeyDown}
      className="relative min-h-0 flex-1 overflow-hidden"
    >
      <div
        ref={attachViewport}
        onPointerDown={startPan}
        onPointerMove={movePan}
        onPointerUp={endPan}
        onPointerCancel={endPan}
        className={cn(
          "absolute inset-0 cursor-grab touch-none overflow-hidden select-none active:cursor-grabbing",
          // The faint grid backdrop of the reference design.
          "bg-[linear-gradient(color-mix(in_oklab,var(--border)_45%,transparent)_1px,transparent_1px),linear-gradient(90deg,color-mix(in_oklab,var(--border)_45%,transparent)_1px,transparent_1px)] bg-size-[56px_56px] bg-center",
          "dark:bg-[linear-gradient(color-mix(in_oklab,var(--border)_10%,transparent)_1px,transparent_1px),linear-gradient(90deg,color-mix(in_oklab,var(--border)_10%,transparent)_1px,transparent_1px)]",
        )}
      >
        <div
          ref={attachGraph}
          className="absolute top-0 left-0 origin-top-left"
          style={{ width: bounds.w, height: bounds.h, opacity: 0 }}
        >
          {/* Group containers — labeled vertical stacks. */}
          {groupBands.map((band) => (
            <div
              key={band.id}
              style={{
                left: band.x,
                top: band.y,
                width: band.w,
                height: band.h,
                animationDelay: `${delayAt(band.x)}s`,
                animationFillMode: "backwards",
              }}
              className="map-fade-enter absolute rounded-[32px] border border-border/60 bg-card/60 [box-shadow:var(--map-card-shadow)] dark:border-transparent dark:bg-card/50"
            >
              <span className="absolute top-4 left-4 text-[10px] font-medium tracking-wide text-muted-foreground uppercase">
                {band.group}
              </span>
              <div
                ref={registerHitRing(band.id)}
                className="pointer-events-none absolute inset-0 rounded-[32px] border opacity-0 transition-[opacity,border-color] duration-500"
                style={{ borderColor: "transparent" }}
              />
            </div>
          ))}

          {/* Edges: soft orthogonal runs, drawing themselves in. */}
          <svg
            aria-hidden
            width={bounds.w}
            height={bounds.h}
            className="pointer-events-none absolute inset-0 overflow-visible"
          >
            {edges.map((e, i) => {
              const d = edgePath(e.points);
              const length = polylineLength(e.points);
              const delay = delayAt(xOf.get(e.from) ?? 0) + 0.25;
              return (
                <g
                  key={i}
                  className={cn(
                    "transition-opacity duration-300",
                    edgeActive(e) ? "opacity-100" : "opacity-15",
                  )}
                >
                  <path
                    d={d}
                    fill="none"
                    stroke={EDGE_STROKE}
                    strokeWidth={1.4}
                    strokeLinecap="round"
                    className="map-edge-draw"
                    style={{
                      strokeDasharray: length,
                      ["--map-edge-len" as string]: `${length}`,
                      animationDelay: `${delay}s`,
                      animationFillMode: "backwards",
                    }}
                  />
                  <path
                    d={arrowHead(e.points)}
                    fill="none"
                    stroke={EDGE_STROKE}
                    strokeWidth={1.4}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="map-fade-enter"
                    style={{
                      animationDelay: `${delay + 0.5}s`,
                      animationFillMode: "backwards",
                    }}
                  />
                </g>
              );
            })}
          </svg>

          {/* Discrete traveling beams — one per edge, sparse and staggered. */}
          {edges.map((e, i) => {
            const sourceKind = nodesById.get(foldedEdges[e.orig[0]].from)?.kind ?? "entry";
            return (
              <BeamPulse
                key={`b${i}`}
                d={edgePath(e.points)}
                length={polylineLength(e.points)}
                color={KIND_META[sourceKind].hex}
                dimmed={!edgeActive(e)}
                targetId={e.to}
                onArrive={flashTarget}
              />
            );
          })}

          {/* Edge labels. Explicit labels always show, where ELK reserved
              room for them; kind-only labels ("reads", "writes") are
              boilerplate that piles up in shared channels, so they surface
              only while the edge is part of a traced flow. */}
          {edges.map((e, i) => {
            const kindLabel = e.orig
              .map((oi) => foldedEdges[oi].kind)
              .find(Boolean);
            const text = e.label ?? kindLabel;
            if (!text) return null;
            const kindOnly = !e.label;
            const inTrace = trace !== null && e.orig.some((oi) => trace.edges.has(oi));
            if (kindOnly && !inTrace) return null;
            const mid = e.labelPos ?? labelAnchor(e.points);
            if (!mid) return null;
            return (
              <span
                key={`l${i}${kindOnly ? "k" : ""}`}
                className={cn(
                  "map-fade-enter absolute -translate-x-1/2 -translate-y-1/2 rounded-full bg-background px-2 py-0.5 whitespace-nowrap",
                  "pointer-events-none text-xs text-muted-foreground/80 transition-opacity duration-300",
                  !edgeActive(e) && "opacity-15",
                )}
                style={{
                  left: mid.x,
                  top: mid.y,
                  animationDelay: kindOnly
                    ? "0s"
                    : `${delayAt(xOf.get(e.from) ?? 0) + 0.6}s`,
                  animationFillMode: "backwards",
                }}
              >
                {text}
              </span>
            );
          })}

          {/* Node cards, in data (reading) order so Tab follows the story. */}
          {nodes.map((node) => {
            const box = positions.get(node.id);
            if (!box) return null;
            return (
              <MapNodeCard
                key={node.id}
                node={node}
                box={box}
                chips={chips.get(node.id)}
                selected={selected === node.id}
                dimmed={!nodeActive(node.id, node.kind)}
                enterDelay={delayAt(box.x, box.y)}
                hitRingRef={registerHitRing(node.id)}
                onSelect={(id) => setSelected((prev) => (prev === id ? null : id))}
              />
            );
          })}
        </div>
      </div>

      {/* Usage hint the section's aria-describedby points at. */}
      <p id={hintId} className="sr-only">
        Interactive architecture map. Tab reaches each component; Enter or
        Space spotlights its downstream flow and opens its details; Escape
        clears; plus, minus, and zero zoom. A text list of every connection
        follows the map.
      </p>

      {/* Browse-mode text equivalent of the full graph (models included). */}
      <ul className="sr-only">
        {map.graph.nodes.map((node) => {
          const outs = map.graph.edges.filter((edge) => edge.from === node.id);
          return (
            <li key={node.id}>
              {node.label} ({KIND_META[node.kind].label})
              {node.sub ? `: ${node.sub}` : ""}.
              {outs.length > 0
                ? ` Connects to ${outs
                    .map(
                      (edge: MapEdge) =>
                        `${labelOf(edge.to)}${edge.label ? ` (${edge.label})` : ""}`,
                    )
                    .join(", ")}.`
                : ""}
            </li>
          );
        })}
      </ul>

      <MapSidePanel map={map} />
      <MapLegend onKindFocus={setKindFocus} />

      {/* Detail popover for the spotlit card — positioned in screen space
          (outside the pan/zoom transform) so it stays readable at any zoom;
          applyTransform() keeps it glued to its card while panning. */}
      {selectedNode && selectedBox ? (
        <div
          key={selectedNode.id}
          ref={attachPopover}
          data-anchor={JSON.stringify(selectedBox)}
          className="absolute"
        >
          <NodeDetail node={selectedNode} left={0} top={0} />
        </div>
      ) : null}

      <div className="pointer-events-auto absolute right-4 bottom-4 z-20 flex items-center gap-1 rounded-full bg-card/70 p-1 backdrop-blur-md [box-shadow:var(--map-card-shadow)]">
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label="Zoom in"
          className="rounded-full"
          onClick={() => zoomAboutCenter(ZOOM_STEP)}
        >
          <ZoomInIcon />
        </Button>
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label="Zoom out"
          className="rounded-full"
          onClick={() => zoomAboutCenter(1 / ZOOM_STEP)}
        >
          <ZoomOutIcon />
        </Button>
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label="Reset view"
          className="rounded-full"
          onClick={() => {
            const viewport = graphRef.current?.parentElement;
            if (viewport instanceof HTMLDivElement) fitToViewport(viewport);
          }}
        >
          <FrameIcon />
        </Button>
      </div>
    </section>
  );
}
