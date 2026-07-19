/**
 * Deterministic layered layout for the curated codebase map — pure geometry,
 * no React and no DOM.
 *
 * This is the "brackets, not columns" algorithm from the implementation plan's
 * Phase 2, and it supersedes spec_codebase_map.md §5.6. Every node keeps its
 * own longest-path rank; a group is a labeled band drawn *behind* its members
 * wherever they land, so a group may span the adjacent columns its members
 * already occupy (Ingestion spans three). That dissolves the §5.6 contradiction
 * where "pull a group into one column" collapsed the very edges the trace
 * feature exists to show — here every edge stays strictly left-to-right.
 *
 * The pipeline is four passes, each deterministic by construction (no
 * randomness, no force simulation, so the same map always renders identically):
 *
 *   1. Rank by the longest path from a source node.
 *   2. Order within each column with the barycenter heuristic — alternating
 *      down/up sweeps to a fixpoint, ties broken by node id.
 *   3. Pull each group's members into a contiguous block within their columns.
 *   4. Place fixed-size boxes and draw cubic-Bézier edges.
 */

import type { CodebaseMap, MapEdge, MapNode } from "./codebase-map";

/** An axis-aligned rectangle — a node's placed box. */
export interface NodeBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** A group's labeled band: the padded union of its members' boxes. */
export interface GroupBand extends NodeBox {
  group: string;
}

/** One edge's rendered path. */
export interface EdgePath {
  edge: MapEdge;
  /** SVG cubic-Bézier path data (`M … C …`). */
  d: string;
}

/** Everything the renderer needs to draw the map once. */
export interface LayoutResult {
  positions: ReadonlyMap<string, NodeBox>;
  edgePaths: ReadonlyArray<EdgePath>;
  groupBands: ReadonlyArray<GroupBand>;
  bounds: { w: number; h: number };
}

// --- Geometry constants (device-independent units; the SVG scales them). ---

/** MapNode.label cap (codebase-map.ts), the driver of node width. */
const LABEL_MAX_CHARS = 28;
/** Assumed average glyph advance at the map's font size. */
const CHAR_WIDTH = 7;
/** Horizontal padding inside a node box. */
const NODE_PADDING_X = 16;
/** Fixed node width — fits the 28-char label cap at the assumed advance. */
const NODE_WIDTH = LABEL_MAX_CHARS * CHAR_WIDTH + 2 * NODE_PADDING_X;
/** Height of a node showing only its label. */
const NODE_HEIGHT_LABEL = 44;
/** Height of a node showing a label and a `sub`. */
const NODE_HEIGHT_SUB = 64;
/** Horizontal space between ranks — leaves room for edge labels. */
const COLUMN_GAP = 120;
/** Vertical space between stacked nodes within a column. */
const ROW_GAP = 28;
/** Canvas padding around the whole drawing. */
const MARGIN = 48;
/** Group-band inset around its members on every side. */
const BAND_PADDING = 18;
/** Extra top room a band reserves for its own label. */
const BAND_LABEL_HEIGHT = 24;
/** Barycenter iteration cap — dot's shape, dagre's fixpoint. */
const MAX_SWEEPS = 24;
/** Horizontal control-point offset as a fraction of an edge's span. */
const EDGE_CURVE = 0.5;
/** Outward bow for a (defensive) same-column edge. */
const SAME_COLUMN_BOW = 48;

/**
 * Longest-path rank of every node: 0 for a source (no inbound edge), otherwise
 * one past its deepest predecessor. Because every edge then strictly increases
 * rank, the drawing is guaranteed left-to-right with no same-rank edges.
 *
 * @param nodes - The graph's nodes.
 * @param edges - The graph's edges (self-edges and danglers are ignored).
 * @returns A node-id → rank map.
 * @throws If the graph contains a cycle. The checked-in data is DAG-guarded
 *   (codebase-map invariant 8), so this only ever fires on a broken fixture.
 */
function computeRanks(
  nodes: readonly MapNode[],
  edges: readonly MapEdge[],
): Map<string, number> {
  const successors = new Map<string, string[]>();
  const indegree = new Map<string, number>();
  for (const node of nodes) {
    successors.set(node.id, []);
    indegree.set(node.id, 0);
  }
  for (const edge of edges) {
    if (edge.from === edge.to) continue;
    const out = successors.get(edge.from);
    if (out === undefined || !indegree.has(edge.to)) continue;
    out.push(edge.to);
    indegree.set(edge.to, (indegree.get(edge.to) ?? 0) + 1);
  }

  const rank = new Map<string, number>();
  const queue: string[] = [];
  for (const node of nodes) {
    rank.set(node.id, 0);
    if ((indegree.get(node.id) ?? 0) === 0) queue.push(node.id);
  }

  // Kahn's topological sweep, relaxing rank as each node is finalized: a node
  // is dequeued only once every predecessor has already relaxed it.
  let processed = 0;
  for (let head = 0; head < queue.length; head += 1) {
    const id = queue[head];
    processed += 1;
    const here = rank.get(id) ?? 0;
    for (const next of successors.get(id) ?? []) {
      if ((rank.get(next) ?? 0) < here + 1) rank.set(next, here + 1);
      const remaining = (indegree.get(next) ?? 0) - 1;
      indegree.set(next, remaining);
      if (remaining === 0) queue.push(next);
    }
  }
  if (processed !== nodes.length) {
    throw new Error("map-layout: the graph has a cycle (expected a DAG)");
  }
  return rank;
}

/** Group node ids into columns by rank, initially in declaration order. */
function columnsByRank(
  nodes: readonly MapNode[],
  rank: ReadonlyMap<string, number>,
): string[][] {
  const maxRank = Math.max(0, ...rank.values());
  const columns: string[][] = Array.from({ length: maxRank + 1 }, () => []);
  for (const node of nodes) {
    columns[rank.get(node.id) ?? 0].push(node.id);
  }
  return columns;
}

/** Each node's current index within its own column. */
function slotIndex(columns: readonly string[][]): Map<string, number> {
  const slot = new Map<string, number>();
  for (const column of columns) {
    column.forEach((id, i) => slot.set(id, i));
  }
  return slot;
}

/**
 * Reorder one column by the mean slot of each node's neighbors in the swept
 * direction. Nodes with no such neighbor keep their current slot; exact ties
 * (and no-neighbor nodes) break by node id, so the sort is deterministic.
 *
 * @returns Whether the column's order actually changed.
 */
function reorderColumn(
  column: string[],
  neighbors: ReadonlyMap<string, string[]>,
  slot: ReadonlyMap<string, number>,
): boolean {
  if (column.length < 2) return false;
  const bary = new Map<string, number>();
  column.forEach((id, index) => {
    let sum = 0;
    let count = 0;
    for (const other of neighbors.get(id) ?? []) {
      const s = slot.get(other);
      if (s !== undefined) {
        sum += s;
        count += 1;
      }
    }
    bary.set(id, count > 0 ? sum / count : index);
  });

  const sorted = [...column].sort((a, b) => {
    const delta = (bary.get(a) ?? 0) - (bary.get(b) ?? 0);
    if (delta !== 0) return delta;
    return a < b ? -1 : a > b ? 1 : 0;
  });

  let changed = false;
  for (let i = 0; i < sorted.length; i += 1) {
    if (sorted[i] !== column[i]) {
      changed = true;
      break;
    }
  }
  if (changed) {
    for (let i = 0; i < sorted.length; i += 1) column[i] = sorted[i];
  }
  return changed;
}

/**
 * Reduce edge crossings with the barycenter heuristic: alternate a down sweep
 * (order each column by its predecessors) and an up sweep (by successors),
 * stopping when a whole sweep changes nothing or after {@link MAX_SWEEPS}. Slots
 * are re-read per column so a sweep sees its own earlier reorderings.
 */
function orderColumns(
  columns: string[][],
  predecessors: ReadonlyMap<string, string[]>,
  successors: ReadonlyMap<string, string[]>,
): void {
  for (let sweep = 0; sweep < MAX_SWEEPS; sweep += 1) {
    const goingDown = sweep % 2 === 0;
    const neighbors = goingDown ? predecessors : successors;
    let changed = false;
    if (goingDown) {
      for (let r = 1; r < columns.length; r += 1) {
        if (reorderColumn(columns[r], neighbors, slotIndex(columns))) changed = true;
      }
    } else {
      for (let r = columns.length - 2; r >= 0; r -= 1) {
        if (reorderColumn(columns[r], neighbors, slotIndex(columns))) changed = true;
      }
    }
    if (!changed) break;
  }
}

/**
 * Pull each group's members into a contiguous block within every column,
 * anchored at the block's first member and preserving the barycenter order
 * among the members. Ungrouped nodes keep their slots, and nodes never move
 * between columns — a group simply becomes a band spanning the columns it
 * already occupies.
 */
function groupContiguous(
  columns: string[][],
  groupOf: ReadonlyMap<string, string | undefined>,
): void {
  for (const column of columns) {
    const emitted = new Set<string>();
    const result: string[] = [];
    for (const id of column) {
      const group = groupOf.get(id);
      if (group === undefined) {
        result.push(id);
      } else if (!emitted.has(group)) {
        emitted.add(group);
        for (const other of column) {
          if (groupOf.get(other) === group) result.push(other);
        }
      }
    }
    for (let i = 0; i < result.length; i += 1) column[i] = result[i];
  }
}

/**
 * Place each node in a fixed-width box. Columns are top-to-bottom stacks with
 * {@link ROW_GAP} between boxes, vertically centered against the tallest column
 * so the drawing reads as a balanced band rather than a ragged top edge.
 */
function placeNodes(
  nodes: readonly MapNode[],
  columns: readonly string[][],
): Map<string, NodeBox> {
  const heightOf = new Map<string, number>(
    nodes.map((node) => [node.id, node.sub ? NODE_HEIGHT_SUB : NODE_HEIGHT_LABEL]),
  );
  const stackHeight = columns.map(
    (column) =>
      column.reduce((sum, id) => sum + (heightOf.get(id) ?? NODE_HEIGHT_LABEL), 0) +
      Math.max(0, column.length - 1) * ROW_GAP,
  );
  const contentHeight = Math.max(0, ...stackHeight);

  const boxes = new Map<string, NodeBox>();
  columns.forEach((column, c) => {
    const x = c * (NODE_WIDTH + COLUMN_GAP);
    let y = (contentHeight - stackHeight[c]) / 2;
    for (const id of column) {
      const h = heightOf.get(id) ?? NODE_HEIGHT_LABEL;
      boxes.set(id, { x, y, w: NODE_WIDTH, h });
      y += h + ROW_GAP;
    }
  });
  return boxes;
}

/** Build each group's band as the padded union of its members' boxes. */
function groupBands(
  nodes: readonly MapNode[],
  boxes: ReadonlyMap<string, NodeBox>,
): GroupBand[] {
  const order: string[] = [];
  const members = new Map<string, NodeBox[]>();
  for (const node of nodes) {
    if (node.group === undefined) continue;
    const box = boxes.get(node.id);
    if (box === undefined) continue;
    const list = members.get(node.group);
    if (list === undefined) {
      members.set(node.group, [box]);
      order.push(node.group);
    } else {
      list.push(box);
    }
  }

  return order.map((group) => {
    const list = members.get(group) ?? [];
    const minX = Math.min(...list.map((b) => b.x));
    const minY = Math.min(...list.map((b) => b.y));
    const maxX = Math.max(...list.map((b) => b.x + b.w));
    const maxY = Math.max(...list.map((b) => b.y + b.h));
    return {
      group,
      x: minX - BAND_PADDING,
      y: minY - BAND_PADDING - BAND_LABEL_HEIGHT,
      w: maxX - minX + 2 * BAND_PADDING,
      h: maxY - minY + 2 * BAND_PADDING + BAND_LABEL_HEIGHT,
    };
  });
}

/**
 * Shift every box and band into the positive quadrant with a {@link MARGIN}
 * border, so bands that overhang the leftmost/topmost nodes are never clipped.
 * Mutates its inputs and returns the resulting canvas size.
 */
function normalize(
  boxes: Map<string, NodeBox>,
  bands: GroupBand[],
): { w: number; h: number } {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  const consider = (x: number, y: number, w: number, h: number): void => {
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x + w);
    maxY = Math.max(maxY, y + h);
  };
  for (const box of boxes.values()) consider(box.x, box.y, box.w, box.h);
  for (const band of bands) consider(band.x, band.y, band.w, band.h);
  if (!Number.isFinite(minX)) return { w: 2 * MARGIN, h: 2 * MARGIN };

  const dx = MARGIN - minX;
  const dy = MARGIN - minY;
  for (const box of boxes.values()) {
    box.x += dx;
    box.y += dy;
  }
  for (const band of bands) {
    band.x += dx;
    band.y += dy;
  }
  return { w: maxX - minX + 2 * MARGIN, h: maxY - minY + 2 * MARGIN };
}

/** Round to two decimals for compact, stable SVG path data (no `-0`). */
function fmt(n: number): string {
  return (Math.round(n * 100) / 100).toString();
}

/**
 * A cubic Bézier from one box to another with horizontal control points, so the
 * curve leaves and enters each node horizontally. Left-to-right edges (the only
 * kind the corrected DAG produces) run right-edge → left-edge; a same-column
 * edge — defensive, since longest-path ranking never yields one — bows outward
 * past the boxes' right edges instead of cutting through them.
 */
function edgeData(from: NodeBox, to: NodeBox): string {
  if (Math.abs(from.x - to.x) < 0.5) {
    const sx = from.x + from.w;
    const sy = from.y + from.h / 2;
    const ex = to.x + to.w;
    const ey = to.y + to.h / 2;
    const bow = Math.max(from.w, to.w) + SAME_COLUMN_BOW;
    return `M ${fmt(sx)} ${fmt(sy)} C ${fmt(sx + bow)} ${fmt(sy)} ${fmt(ex + bow)} ${fmt(ey)} ${fmt(ex)} ${fmt(ey)}`;
  }

  const leftToRight = from.x < to.x;
  const start = leftToRight
    ? { x: from.x + from.w, y: from.y + from.h / 2 }
    : { x: from.x, y: from.y + from.h / 2 };
  const end = leftToRight
    ? { x: to.x, y: to.y + to.h / 2 }
    : { x: to.x + to.w, y: to.y + to.h / 2 };
  const curve = Math.abs(end.x - start.x) * EDGE_CURVE;
  const dir = leftToRight ? 1 : -1;
  const c1x = start.x + dir * curve;
  const c2x = end.x - dir * curve;
  return `M ${fmt(start.x)} ${fmt(start.y)} C ${fmt(c1x)} ${fmt(start.y)} ${fmt(c2x)} ${fmt(end.y)} ${fmt(end.x)} ${fmt(end.y)}`;
}

/**
 * Lay out the whole map deterministically.
 *
 * @param map - The curated codebase map.
 * @returns Node boxes, group bands, edge paths, and the canvas size.
 * @throws If the graph is not a DAG (see {@link computeRanks}).
 */
export function layout(map: CodebaseMap): LayoutResult {
  const { nodes, edges } = map.graph;
  const rank = computeRanks(nodes, edges);

  const predecessors = new Map<string, string[]>();
  const successors = new Map<string, string[]>();
  for (const node of nodes) {
    predecessors.set(node.id, []);
    successors.set(node.id, []);
  }
  for (const edge of edges) {
    if (edge.from === edge.to) continue;
    const out = successors.get(edge.from);
    const into = predecessors.get(edge.to);
    if (out === undefined || into === undefined) continue;
    out.push(edge.to);
    into.push(edge.from);
  }

  const columns = columnsByRank(nodes, rank);
  orderColumns(columns, predecessors, successors);
  groupContiguous(
    columns,
    new Map(nodes.map((node) => [node.id, node.group])),
  );

  const positions = placeNodes(nodes, columns);
  const bands = groupBands(nodes, positions);
  const bounds = normalize(positions, bands);

  const edgePaths: EdgePath[] = [];
  for (const edge of edges) {
    const from = positions.get(edge.from);
    const to = positions.get(edge.to);
    if (from === undefined || to === undefined) continue;
    edgePaths.push({ edge, d: edgeData(from, to) });
  }

  return { positions, edgePaths, groupBands: bands, bounds };
}
