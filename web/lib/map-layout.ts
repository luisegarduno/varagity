/**
 * Deterministic flow-map layout for the codebase map, powered by ELK's layered
 * algorithm with orthogonal edge routing — ported from foglamp's renderer
 * (foglamp-labs/foglamp `apps/web/src/components/scan/layout.ts` +
 * `fold-graph.ts`, Apache-2.0) so `/map` lays out exactly like
 * foglamp.dev's scan viewer does.
 *
 * The pipeline:
 *
 *   1. **Fold** — `model`/`tool` nodes disappear into on-card chips of the
 *      nodes that use them; edges to folded nodes are dropped. This is what
 *      removes the model fan-in noise.
 *   2. **Pass A** — each group's members are laid out top-to-bottom in
 *      isolation (ELK layered, DOWN).
 *   3. **Pass B** — the root graph (ungrouped nodes + group boxes) lays out
 *      left-to-right with BRANDES_KOEPF/BALANCED placement and orthogonal
 *      routing; ELK reserves room for inline edge labels while routing, so
 *      labels can never collide. Cross-group edges attach to the group
 *      container and dedupe per endpoint pair (`orig` keeps the originals).
 *   4. **Row snap** aligns items whose centers almost agree; the
 *      **empty-band squeeze** compresses void bands the router left behind.
 *
 * Same input → same coordinates (ELK layered is deterministic). Async: elkjs
 * loads lazily via `elk-api` + its fake in-process worker, which survives
 * bundlers and runs in Node and the browser alike.
 */

import type { ELK, ElkNode } from "elkjs/lib/elk-api";

import type { CodebaseMap, MapEdge, MapNode } from "./codebase-map";

// elkjs via elk-api + the fake worker instead of elk.bundled.js: the bundle's
// own `require("./elk-worker.min.js")` doesn't survive Next bundling, while an
// explicit workerFactory works everywhere. Module shapes are probed because
// CJS interop differs per bundler (module, .default, .default.default).
let elkInstance: ELK | null = null;
async function getElk(): Promise<ELK> {
  if (!elkInstance) {
    const [elkApi, worker] = (await Promise.all([
      import("elkjs/lib/elk-api.js"),
      // @ts-expect-error — no types for the worker entry
      import("elkjs/lib/elk-worker.min.js"),
    ])) as [
      { default: (new (opts: object) => ELK) | { default: new (opts: object) => ELK } },
      Record<string, unknown>,
    ];
    const ELKCtor = (
      typeof elkApi.default === "function" ? elkApi.default : elkApi.default.default
    ) as new (opts: object) => ELK;
    const workerNs = worker as { Worker?: unknown; default?: { Worker?: unknown } };
    const FakeWorker = (workerNs.Worker ??
      workerNs.default?.Worker ??
      workerNs.default) as new (url?: string) => Worker;
    if (typeof FakeWorker !== "function") {
      throw new Error("elk-worker module shape not recognized");
    }
    elkInstance = new ELKCtor({
      workerFactory: (url?: string) => new FakeWorker(url),
    });
  }
  return elkInstance;
}

// --- Folding (foglamp's fold-graph) ----------------------------------------

/** One model/tool chip folded into a card. */
export interface ModelChip {
  label: string;
  domain?: string;
}

/** The renderable view of a map after folding models/tools into chips. */
export interface CondensedMap {
  /** Nodes that render as cards, in data order. */
  nodes: MapNode[];
  /** Structural edges between rendered cards. */
  edges: MapEdge[];
  /** Card id → chips, in first-edge order (models sort before tools). */
  chips: Map<string, ModelChip[]>;
}

const FOLDED_KINDS = new Set<MapNode["kind"]>(["model", "tool"]);

/**
 * Fold every `model`/`tool` node into chips on the cards that point at it —
 * the foglamp treatment: model usage reads as a badge on the consumer, not as
 * fan-in arrows. Edges touching folded nodes are dropped; structural edges
 * are kept.
 *
 * @param map - The curated map.
 * @returns The renderable nodes/edges plus the per-card chip lists.
 */
export function condense(map: CodebaseMap): CondensedMap {
  const byId = new Map(map.graph.nodes.map((node) => [node.id, node]));
  const chips = new Map<string, ModelChip[]>();

  for (const edge of map.graph.edges) {
    const target = byId.get(edge.to);
    const source = byId.get(edge.from);
    if (!target || !source || !FOLDED_KINDS.has(target.kind)) continue;
    const list = chips.get(source.id) ?? [];
    if (!list.some((chip) => chip.label === target.label)) {
      list.push({ label: target.label, domain: target.domain });
    }
    chips.set(source.id, list);
  }

  const nodes = map.graph.nodes.filter((node) => !FOLDED_KINDS.has(node.kind));
  const alive = new Set(nodes.map((node) => node.id));
  const edges = map.graph.edges.filter(
    (edge) => alive.has(edge.from) && alive.has(edge.to),
  );
  return { nodes, edges, chips };
}

// --- Card sizing (matches components/map/MapNode.tsx) ----------------------

/** Header row height — a card with no chips is exactly this tall. */
const HEAD_H = 56;
/** Extra height per chip row (16px row + 8px gap). */
const CHIP_ROW_H = 24;
/** Base card width; hubs grow with connection degree so they read bigger. */
const BASE_W = 208;
const DEGREE_W = 7;
const DEGREE_CAP = 6;

/** Chips stack one per row, so a card grows by one row per chip. */
function nodeHeight(chipCount: number): number {
  if (chipCount === 0) return HEAD_H;
  return HEAD_H + chipCount * CHIP_ROW_H + 12;
}

// --- Geometry types --------------------------------------------------------

/** An axis-aligned rectangle — a node's placed box. */
export interface NodeBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** A group's container: the box drawn behind (and labeled above) its members. */
export interface GroupBand extends NodeBox {
  /** Stable id (`group:N`) — rendered edges may attach to it. */
  id: string;
  group: string;
}

/** One routed edge of the drawing. */
export interface RenderedEdge {
  /** Endpoint ids — node ids, or a group id for edges attached to a stack. */
  from: string;
  to: string;
  label?: string;
  /** Orthogonal polyline in canvas coordinates. */
  points: { x: number; y: number }[];
  /** ELK-placed label center; ELK reserved room for it while routing. */
  labelPos?: { x: number; y: number };
  /** Indices into the condensed edge list this rendered edge represents. */
  orig: number[];
}

/** Everything the renderer needs to draw the map once. */
export interface MapLayout {
  /** The renderable nodes (models/tools folded), in data order. */
  nodes: ReadonlyArray<MapNode>;
  /** The structural (condensed) edges `RenderedEdge.orig` indexes into. */
  foldedEdges: ReadonlyArray<MapEdge>;
  positions: ReadonlyMap<string, NodeBox>;
  groupBands: ReadonlyArray<GroupBand>;
  edges: ReadonlyArray<RenderedEdge>;
  /** Renderable node id → the chips folded into its card. */
  chips: ReadonlyMap<string, ModelChip[]>;
  bounds: { w: number; h: number };
}

/** Label box ELK reserves while routing — mirrors the rendered pill
 *  (text-xs ≈ 6px/char + px-2 padding, py-0.5 → 22px tall). */
const LABEL_H = 22;
function labelDims(text: string): { width: number; height: number } {
  return { width: Math.round(text.length * 6) + 14, height: LABEL_H };
}

const GROUP_PAD = { top: 46, right: 16, bottom: 16, left: 16 };

/**
 * Lay out the whole map, foglamp-style.
 *
 * @param map - The curated codebase map.
 * @returns Placed nodes, group containers, chip lists, routed edges, and the
 *   canvas size.
 */
export async function layoutGraph(map: CodebaseMap): Promise<MapLayout> {
  const folded = condense(map);
  const elk = await getElk();

  // Degree-scaled card sizes: hubs read bigger.
  const degree = new Map<string, number>();
  for (const e of folded.edges) {
    degree.set(e.from, (degree.get(e.from) ?? 0) + 1);
    degree.set(e.to, (degree.get(e.to) ?? 0) + 1);
  }
  const sizeOf = new Map(
    folded.nodes.map((n) => [
      n.id,
      {
        width: BASE_W + Math.min(degree.get(n.id) ?? 0, DEGREE_CAP) * DEGREE_W,
        height: nodeHeight(folded.chips.get(n.id)?.length ?? 0),
      },
    ]),
  );
  const nodeById = new Map(folded.nodes.map((n) => [n.id, n]));

  // Group membership, in order of first appearance.
  const groupNames: string[] = [];
  const membersByGroup = new Map<string, MapNode[]>();
  for (const n of folded.nodes) {
    if (!n.group) continue;
    if (!membersByGroup.has(n.group)) {
      membersByGroup.set(n.group, []);
      groupNames.push(n.group);
    }
    membersByGroup.get(n.group)?.push(n);
  }
  const groupIdOf = (name: string): string => `group:${groupNames.indexOf(name)}`;
  const groupOfNode = (id: string): string | undefined => nodeById.get(id)?.group;

  // ── Pass A: each group in isolation, top-to-bottom ────────────────────────
  const groupLayouts = new Map<
    string,
    {
      size: { width: number; height: number };
      children: Map<string, { x: number; y: number }>;
      edges: {
        points: { x: number; y: number }[];
        labelPos?: { x: number; y: number };
        origIndex: number;
      }[];
    }
  >();

  for (const name of groupNames) {
    const members = membersByGroup.get(name) ?? [];
    const memberIds = new Set(members.map((m) => m.id));
    const internal = folded.edges
      .map((e, i) => ({ e, i }))
      .filter(({ e }) => memberIds.has(e.from) && memberIds.has(e.to));

    const input: ElkNode = {
      id: "root",
      layoutOptions: {
        "elk.algorithm": "layered",
        "elk.direction": "DOWN",
        "elk.edgeRouting": "ORTHOGONAL",
        "elk.layered.spacing.nodeNodeBetweenLayers": "30",
        "elk.spacing.nodeNode": "18",
        "elk.spacing.edgeNode": "14",
        "elk.edgeLabels.inline": "true",
        "elk.spacing.edgeLabel": "4",
        "elk.padding": `[top=${GROUP_PAD.top},left=${GROUP_PAD.left},bottom=${GROUP_PAD.bottom},right=${GROUP_PAD.right}]`,
      },
      children: members.map((m) => ({
        id: m.id,
        width: sizeOf.get(m.id)?.width,
        height: sizeOf.get(m.id)?.height,
      })),
      edges: internal.map(({ e, i }) => ({
        id: `e${i}`,
        sources: [e.from],
        targets: [e.to],
        ...(e.label
          ? { labels: [{ id: `el${i}`, text: e.label, ...labelDims(e.label) }] }
          : {}),
      })),
    };
    const res = await elk.layout(input);
    const children = new Map<string, { x: number; y: number }>();
    for (const c of res.children ?? []) children.set(c.id, { x: c.x ?? 0, y: c.y ?? 0 });
    const groupEdges = (res.edges ?? []).map((el) => {
      const sec = el.sections?.[0];
      const lbl = el.labels?.[0];
      return {
        origIndex: Number(el.id.slice(1)),
        points: sec ? [sec.startPoint, ...(sec.bendPoints ?? []), sec.endPoint] : [],
        labelPos:
          lbl && lbl.x != null && lbl.y != null
            ? { x: lbl.x + (lbl.width ?? 0) / 2, y: lbl.y + (lbl.height ?? 0) / 2 }
            : undefined,
      };
    });
    groupLayouts.set(name, {
      size: { width: res.width ?? 0, height: res.height ?? 0 },
      children,
      edges: groupEdges,
    });
  }

  // ── Pass B: root graph — ungrouped nodes + group boxes, left-to-right ────
  // Cross-group edges are remapped to the group box and deduped.
  interface RootEdge {
    from: string;
    to: string;
    label?: string;
    orig: number[];
  }
  const rootEdges = new Map<string, RootEdge>();
  folded.edges.forEach((e, i) => {
    const gFrom = groupOfNode(e.from);
    const gTo = groupOfNode(e.to);
    if (gFrom && gFrom === gTo) return; // internal — handled in pass A
    const from = gFrom ? groupIdOf(gFrom) : e.from;
    const to = gTo ? groupIdOf(gTo) : e.to;
    const key = `${from}→${to}`;
    const cur = rootEdges.get(key);
    if (cur) {
      cur.orig.push(i);
      if (!cur.label && e.label) cur.label = e.label;
    } else {
      rootEdges.set(key, { from, to, label: e.label, orig: [i] });
    }
  });

  const ungrouped = folded.nodes.filter((n) => !n.group);
  const rootChildren: ElkNode[] = [
    ...ungrouped.map((n) => ({
      id: n.id,
      width: sizeOf.get(n.id)?.width,
      height: sizeOf.get(n.id)?.height,
    })),
    ...groupNames.map((name) => ({
      id: groupIdOf(name),
      width: groupLayouts.get(name)?.size.width,
      height: groupLayouts.get(name)?.size.height,
    })),
  ];
  const dense = rootChildren.length > 12;
  const rootInput: ElkNode = {
    id: "root",
    layoutOptions: {
      "elk.algorithm": "layered",
      "elk.direction": "RIGHT",
      "elk.edgeRouting": "ORTHOGONAL",
      "elk.layered.mergeEdges": "true",
      // Inline labels reserve their own room mid-edge, so layer gaps stay
      // modest instead of carrying the labels.
      "elk.layered.spacing.nodeNodeBetweenLayers": dense ? "56" : "72",
      "elk.spacing.nodeNode": dense ? "18" : "26",
      "elk.spacing.edgeNode": dense ? "16" : "24",
      "elk.spacing.edgeEdge": "14",
      // BRANDES_KOEPF/BALANCED packs rows far tighter than NETWORK_SIMPLEX,
      // which trades area for straight edges and spreads big maps into
      // sparse, bureaucratic grids. Post-compaction then pulls layers together.
      "elk.layered.nodePlacement.strategy": "BRANDES_KOEPF",
      "elk.layered.nodePlacement.bk.fixedAlignment": "BALANCED",
      "elk.layered.compaction.postCompaction.strategy": "EDGE_LENGTH",
      "elk.edgeLabels.inline": "true",
      "elk.spacing.edgeLabel": "4",
      "elk.padding": "[top=16,left=16,bottom=16,right=16]",
    },
    children: rootChildren,
    edges: [...rootEdges.values()].map((e, i) => ({
      id: `r${i}`,
      sources: [e.from],
      targets: [e.to],
      ...(e.label
        ? { labels: [{ id: `rl${i}`, text: e.label, ...labelDims(e.label) }] }
        : {}),
    })),
  };
  const rootRes = await elk.layout(rootInput);
  const rootPos = new Map<string, { x: number; y: number }>();
  for (const c of rootRes.children ?? []) rootPos.set(c.id, { x: c.x ?? 0, y: c.y ?? 0 });

  // ── Row snap ──────────────────────────────────────────────────────────────
  // ELK's placement often leaves items that read as one row a few px apart.
  // Cluster root-level items whose vertical centers are within SNAP_TOL and
  // align each cluster to its mean center. Edge endpoints shift to match,
  // keeping orthogonal routes orthogonal.
  const SNAP_TOL = 20;
  const rowDelta = new Map<string, number>();
  {
    const items = rootChildren
      .map((c) => {
        const pos = rootPos.get(c.id);
        return pos
          ? { id: c.id, centerY: pos.y + (c.height ?? 0) / 2 }
          : null;
      })
      .filter((i): i is NonNullable<typeof i> => i !== null)
      .sort((a, b) => a.centerY - b.centerY);
    let cluster: typeof items = [];
    const flush = (): void => {
      if (cluster.length < 2) return;
      const mean = cluster.reduce((s, i) => s + i.centerY, 0) / cluster.length;
      for (const i of cluster) {
        const dy = mean - i.centerY;
        if (dy === 0) continue;
        rowDelta.set(i.id, dy);
        const pos = rootPos.get(i.id);
        if (pos) rootPos.set(i.id, { x: pos.x, y: pos.y + dy });
      }
    };
    for (const item of items) {
      if (
        cluster.length > 0 &&
        item.centerY - cluster[cluster.length - 1].centerY <= SNAP_TOL
      ) {
        cluster.push(item);
      } else {
        flush();
        cluster = [item];
      }
    }
    flush();
  }

  /** Shift a root edge's endpoints by their nodes' snap deltas, preserving
   *  orthogonality: the endpoint's adjoining horizontal stub moves with it. */
  function snapEdgePoints(
    points: { x: number; y: number }[],
    fromId: string,
    toId: string,
  ): { x: number; y: number }[] {
    const dS = rowDelta.get(fromId) ?? 0;
    const dT = rowDelta.get(toId) ?? 0;
    if ((dS === 0 && dT === 0) || points.length < 2) return points;
    const p = points.map((pt) => ({ ...pt }));
    const n = p.length;
    const horizontal = (a: { y: number }, b: { y: number }): boolean =>
      Math.abs(a.y - b.y) < 0.5;
    p[0].y += dS;
    p[n - 1].y += dT;
    if (n === 2) return p;
    if (horizontal(points[0], points[1]) && n > 3) p[1].y += dS;
    if (horizontal(points[n - 2], points[n - 1]) && n > 3) p[n - 2].y += dT;
    if (n === 3) {
      if (horizontal(points[0], points[1])) p[1].y += dS;
      else if (horizontal(points[1], points[2])) p[1].y += dT;
    }
    return p;
  }

  // ── Compose absolute coordinates ──────────────────────────────────────────
  const positions = new Map<string, NodeBox>();
  for (const n of ungrouped) {
    const p = rootPos.get(n.id) ?? { x: 0, y: 0 };
    const s = sizeOf.get(n.id) ?? { width: BASE_W, height: HEAD_H };
    positions.set(n.id, { x: p.x, y: p.y, w: s.width, h: s.height });
  }
  const groupBands: GroupBand[] = [];
  for (const name of groupNames) {
    const gl = groupLayouts.get(name);
    if (!gl) continue;
    const origin = rootPos.get(groupIdOf(name)) ?? { x: 0, y: 0 };
    groupBands.push({
      id: groupIdOf(name),
      group: name,
      x: origin.x,
      y: origin.y,
      w: gl.size.width,
      h: gl.size.height,
    });
    for (const m of membersByGroup.get(name) ?? []) {
      const rel = gl.children.get(m.id) ?? { x: 0, y: 0 };
      const s = sizeOf.get(m.id) ?? { width: BASE_W, height: HEAD_H };
      positions.set(m.id, {
        x: origin.x + rel.x,
        y: origin.y + rel.y,
        w: s.width,
        h: s.height,
      });
    }
  }

  const rendered: RenderedEdge[] = [];
  // Internal group edges, offset to absolute space.
  for (const name of groupNames) {
    const gl = groupLayouts.get(name);
    if (!gl) continue;
    const origin = rootPos.get(groupIdOf(name)) ?? { x: 0, y: 0 };
    for (const ge of gl.edges) {
      const orig = folded.edges[ge.origIndex];
      rendered.push({
        from: orig.from,
        to: orig.to,
        label: orig.label,
        orig: [ge.origIndex],
        points: ge.points.map((p) => ({ x: p.x + origin.x, y: p.y + origin.y })),
        labelPos: ge.labelPos
          ? { x: ge.labelPos.x + origin.x, y: ge.labelPos.y + origin.y }
          : undefined,
      });
    }
  }
  // Root edges (already absolute — flat root graph), endpoints snapped along
  // with their nodes.
  const rootEdgeList = [...rootEdges.values()];
  (rootRes.edges ?? []).forEach((el, i) => {
    const spec = rootEdgeList[i];
    const sec = el.sections?.[0];
    const points = sec ? [sec.startPoint, ...(sec.bendPoints ?? []), sec.endPoint] : [];
    const lbl = el.labels?.[0];
    const labelDy = ((rowDelta.get(spec.from) ?? 0) + (rowDelta.get(spec.to) ?? 0)) / 2;
    rendered.push({
      from: spec.from,
      to: spec.to,
      label: spec.label,
      orig: spec.orig,
      points: snapEdgePoints(points, spec.from, spec.to),
      labelPos:
        lbl && lbl.x != null && lbl.y != null
          ? {
              x: lbl.x + (lbl.width ?? 0) / 2,
              y: lbl.y + (lbl.height ?? 0) / 2 + labelDy,
            }
          : undefined,
    });
  });

  // ── Empty-band squeeze ────────────────────────────────────────────────────
  // Any interior band (horizontal or vertical) that contains no nodes, groups,
  // labels, or parallel edge runs is compressed to BAND_KEEP. The remap is
  // piecewise-linear and monotonic per axis, so orthogonal routes stay
  // orthogonal and nothing can overlap that didn't before.
  {
    const xOcc: Interval[] = [];
    const yOcc: Interval[] = [];
    for (const box of positions.values()) {
      xOcc.push([box.x, box.x + box.w]);
      yOcc.push([box.y, box.y + box.h]);
    }
    for (const g of groupBands) {
      xOcc.push([g.x, g.x + g.w]);
      yOcc.push([g.y, g.y + g.h]);
    }
    for (const e of rendered) {
      if (e.label && e.labelPos) {
        const dims = labelDims(e.label);
        xOcc.push([e.labelPos.x - dims.width / 2, e.labelPos.x + dims.width / 2]);
        yOcc.push([e.labelPos.y - dims.height / 2, e.labelPos.y + dims.height / 2]);
      }
      // A run keeps a small halo so parallel runs in a squeezed channel don't
      // collapse onto each other; runs perpendicular to the squeeze axis are
      // exactly what we want to shorten, so they don't count as occupied.
      for (let i = 0; i < e.points.length - 1; i++) {
        const a = e.points[i];
        const b = e.points[i + 1];
        if (Math.abs(a.y - b.y) < 0.5) yOcc.push([a.y - 7, a.y + 7]);
        if (Math.abs(a.x - b.x) < 0.5) xOcc.push([a.x - 7, a.x + 7]);
      }
    }
    const fx = buildBandRemap(xOcc);
    const fy = buildBandRemap(yOcc);
    for (const box of positions.values()) {
      box.x = fx(box.x);
      box.y = fy(box.y);
    }
    for (const g of groupBands) {
      g.x = fx(g.x);
      g.y = fy(g.y);
    }
    for (const e of rendered) {
      for (const p of e.points) {
        p.x = fx(p.x);
        p.y = fy(p.y);
      }
      if (e.labelPos) {
        e.labelPos.x = fx(e.labelPos.x);
        e.labelPos.y = fy(e.labelPos.y);
      }
    }
  }

  let maxX = 0;
  let maxY = 0;
  for (const box of positions.values()) {
    maxX = Math.max(maxX, box.x + box.w);
    maxY = Math.max(maxY, box.y + box.h);
  }
  for (const g of groupBands) {
    maxX = Math.max(maxX, g.x + g.w);
    maxY = Math.max(maxY, g.y + g.h);
  }

  return {
    nodes: folded.nodes,
    foldedEdges: folded.edges,
    positions,
    groupBands,
    edges: rendered,
    chips: folded.chips,
    bounds: { w: maxX + 16, h: maxY + 16 },
  };
}

type Interval = [number, number];

/** How much of an empty band survives the squeeze — matches the tightest
 *  between-layer spacing so squeezed voids read like normal gaps. */
const BAND_KEEP = 72;

/**
 * Build a monotonic piecewise-linear remap that compresses every gap between
 * occupied intervals down to BAND_KEEP. Positions inside a squeezed gap map
 * proportionally; positions past it shift by the accumulated savings.
 */
function buildBandRemap(occupied: Interval[]): (v: number) => number {
  const sorted = occupied.filter(([a, b]) => b > a).sort((a, b) => a[0] - b[0]);
  const merged: Interval[] = [];
  for (const iv of sorted) {
    const last = merged[merged.length - 1];
    if (last && iv[0] <= last[1]) last[1] = Math.max(last[1], iv[1]);
    else merged.push([iv[0], iv[1]]);
  }
  const cuts: { start: number; end: number }[] = [];
  for (let i = 0; i < merged.length - 1; i++) {
    const start = merged[i][1];
    const end = merged[i + 1][0];
    if (end - start > BAND_KEEP + 8) cuts.push({ start, end });
  }
  if (cuts.length === 0) return (v) => v;
  return (v) => {
    let out = v;
    for (const c of cuts) {
      const len = c.end - c.start;
      if (v >= c.end) out -= len - BAND_KEEP;
      else if (v > c.start) out -= (v - c.start) * (1 - BAND_KEEP / len);
    }
    return out;
  };
}

/**
 * A small arrowhead "V" path at the target end of a polyline — a plain path
 * (not an SVG marker) so its stroke inherits the edge styling directly.
 */
export function arrowHead(points: { x: number; y: number }[], len = 7): string {
  if (points.length < 2) return "";
  const p = points[points.length - 1];
  const q = points[points.length - 2];
  const ang = Math.atan2(p.y - q.y, p.x - q.x);
  const spread = 0.46;
  const a1x = p.x - len * Math.cos(ang - spread);
  const a1y = p.y - len * Math.sin(ang - spread);
  const a2x = p.x - len * Math.cos(ang + spread);
  const a2y = p.y - len * Math.sin(ang + spread);
  return `M ${a1x} ${a1y} L ${p.x} ${p.y} L ${a2x} ${a2y}`;
}

/**
 * Where to hang a fallback edge label: the midpoint of the polyline's LONGEST
 * segment — the open channel run where a label has room (the naive middle
 * point lands labels on top of nodes).
 */
export function labelAnchor(points: { x: number; y: number }[]): {
  x: number;
  y: number;
} | null {
  if (points.length === 0) return null;
  if (points.length === 1) return points[0];
  let best = 0;
  let bestLen = -1;
  for (let i = 0; i < points.length - 1; i++) {
    const len = Math.hypot(points[i + 1].x - points[i].x, points[i + 1].y - points[i].y);
    if (len > bestLen) {
      bestLen = len;
      best = i;
    }
  }
  return {
    x: (points[best].x + points[best + 1].x) / 2,
    y: (points[best].y + points[best + 1].y) / 2,
  };
}

/** Orthogonal polyline → SVG path with rounded corners. The generous default
 *  radius turns right angles into soft S-curves; it self-clamps to half of
 *  each adjoining segment, so short zig-zags degrade gracefully. */
export function edgePath(points: { x: number; y: number }[], r = 56): string {
  if (points.length === 0) return "";
  if (points.length < 3) {
    return points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`).join(" ");
  }
  const dist = (a: { x: number; y: number }, b: { x: number; y: number }): number =>
    Math.hypot(b.x - a.x, b.y - a.y);
  const toward = (
    from: { x: number; y: number },
    to: { x: number; y: number },
    d: number,
  ): { x: number; y: number } => {
    const len = dist(from, to) || 1;
    return {
      x: from.x + ((to.x - from.x) / len) * d,
      y: from.y + ((to.y - from.y) / len) * d,
    };
  };

  let d = `M ${points[0].x} ${points[0].y}`;
  for (let i = 1; i < points.length - 1; i++) {
    const prev = points[i - 1];
    const p = points[i];
    const next = points[i + 1];
    const r1 = Math.min(r, dist(prev, p) / 2);
    const r2 = Math.min(r, dist(p, next) / 2);
    const a = toward(p, prev, r1);
    const b = toward(p, next, r2);
    d += ` L ${a.x} ${a.y} Q ${p.x} ${p.y} ${b.x} ${b.y}`;
  }
  const last = points[points.length - 1];
  d += ` L ${last.x} ${last.y}`;
  return d;
}

/** Total length of a polyline in px — drives beam travel time. */
export function polylineLength(points: { x: number; y: number }[]): number {
  let length = 0;
  for (let i = 0; i < points.length - 1; i++) {
    length += Math.hypot(points[i + 1].x - points[i].x, points[i + 1].y - points[i].y);
  }
  return length;
}

// --- Precomputed snapshot ---------------------------------------------------
// The map data is static, so the ELK pass runs at authoring time and ships as
// `codebase-map.layout.json` (the drift-guarded `openapi.json` pattern): the
// page imports the snapshot synchronously — instant hydration, and elkjs
// never loads in the browser. `map-layout.test.ts` re-runs ELK and fails on
// drift; regenerate with `bun run test -u`.

/** The JSON-safe shape of {@link MapLayout} (Maps flattened to entries). */
export interface SerializedMapLayout {
  nodes: MapNode[];
  foldedEdges: MapEdge[];
  positions: [string, NodeBox][];
  groupBands: GroupBand[];
  edges: RenderedEdge[];
  chips: [string, ModelChip[]][];
  bounds: { w: number; h: number };
}

/** Flatten a layout for the checked-in JSON snapshot. */
export function serializeLayout(layout: MapLayout): SerializedMapLayout {
  return {
    nodes: [...layout.nodes],
    foldedEdges: [...layout.foldedEdges],
    positions: [...layout.positions.entries()],
    groupBands: [...layout.groupBands],
    edges: [...layout.edges],
    chips: [...layout.chips.entries()],
    bounds: layout.bounds,
  };
}

/** Rehydrate the checked-in snapshot into a {@link MapLayout}. */
export function deserializeLayout(snapshot: SerializedMapLayout): MapLayout {
  return {
    nodes: snapshot.nodes,
    foldedEdges: snapshot.foldedEdges,
    positions: new Map(snapshot.positions),
    groupBands: snapshot.groupBands,
    edges: snapshot.edges,
    chips: new Map(snapshot.chips),
    bounds: snapshot.bounds,
  };
}
