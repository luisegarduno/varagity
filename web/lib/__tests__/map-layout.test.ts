import { describe, expect, it } from "vitest";

import { CODEBASE_MAP } from "@/lib/codebase-map.data";
import type { CodebaseMap } from "@/lib/codebase-map";
import { layout, type NodeBox } from "@/lib/map-layout";

interface Point {
  x: number;
  y: number;
}

/** A layout-only fixture: a group that spans two columns, a long edge, and an
 * ungrouped node sharing a column with group members (to exercise contiguity).
 * It need not satisfy validateMap — layout() trusts its input. */
const FIXTURE: CodebaseMap = {
  project: { name: "Fixture", date: "2026-01-01", summary: "layout fixture" },
  topModels: [],
  topTools: [],
  topIntegrations: [],
  graph: {
    nodes: [
      { id: "a", label: "A", kind: "entry", sub: "source" },
      { id: "b", label: "B", kind: "entry" },
      { id: "g1", label: "G1", kind: "tool", group: "Group" },
      { id: "g2", label: "G2", kind: "tool", group: "Group" },
      { id: "g3", label: "G3", kind: "tool", group: "Group" },
      { id: "mid", label: "MID", kind: "service" },
      { id: "z", label: "Z", kind: "store" },
    ],
    edges: [
      { from: "a", to: "g1" },
      { from: "a", to: "mid" },
      { from: "a", to: "z" }, // long edge: rank 0 -> rank 3
      { from: "b", to: "g2" },
      { from: "b", to: "mid" },
      { from: "mid", to: "g3" },
      { from: "g1", to: "z" },
      { from: "g2", to: "z" },
      { from: "g3", to: "z" },
    ],
  },
};

const CYCLE: CodebaseMap = {
  project: { name: "Cycle", date: "2026-01-01", summary: "cyclic fixture" },
  topModels: [],
  topTools: [],
  topIntegrations: [],
  graph: {
    nodes: [
      { id: "c1", label: "C1", kind: "service" },
      { id: "c2", label: "C2", kind: "service" },
      { id: "c3", label: "C3", kind: "service" },
    ],
    edges: [
      { from: "c1", to: "c2" },
      { from: "c2", to: "c3" },
      { from: "c3", to: "c1" },
    ],
  },
};

function boxFor(positions: ReadonlyMap<string, NodeBox>, id: string): NodeBox {
  const box = positions.get(id);
  if (box === undefined) throw new Error(`no box for "${id}"`);
  return box;
}

function overlaps(a: NodeBox, b: NodeBox): boolean {
  return a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
}

/** Column index per node, derived from x — one distinct x per rank by
 * construction, so equal x means the same column. */
function columnIndex(positions: ReadonlyMap<string, NodeBox>): Map<string, number> {
  const xs = [...new Set([...positions.values()].map((box) => box.x))].sort(
    (p, q) => p - q,
  );
  const indexOfX = new Map(xs.map((x, i) => [x, i]));
  const out = new Map<string, number>();
  for (const [id, box] of positions) out.set(id, indexOfX.get(box.x) ?? 0);
  return out;
}

/** The first and last coordinate pairs of an `M … C …` path. */
function endpoints(d: string): { start: Point; end: Point } {
  const nums = (d.match(/-?\d+(?:\.\d+)?/g) ?? []).map(Number);
  return {
    start: { x: nums[0], y: nums[1] },
    end: { x: nums[nums.length - 2], y: nums[nums.length - 1] },
  };
}

function onBoundary(p: Point, box: NodeBox): boolean {
  const eps = 0.05;
  const within = (v: number, lo: number, hi: number): boolean =>
    v >= lo - eps && v <= hi + eps;
  const onVerticalEdge =
    (Math.abs(p.x - box.x) <= eps || Math.abs(p.x - (box.x + box.w)) <= eps) &&
    within(p.y, box.y, box.y + box.h);
  const onHorizontalEdge =
    (Math.abs(p.y - box.y) <= eps || Math.abs(p.y - (box.y + box.h)) <= eps) &&
    within(p.x, box.x, box.x + box.w);
  return onVerticalEdge || onHorizontalEdge;
}

/** Every group present in the graph, in first-appearance order. */
function groupNames(map: CodebaseMap): string[] {
  const seen: string[] = [];
  for (const node of map.graph.nodes) {
    if (node.group !== undefined && !seen.includes(node.group)) seen.push(node.group);
  }
  return seen;
}

/** The invariants that must hold for any DAG the layout accepts. */
function assertLayoutProperties(map: CodebaseMap): void {
  const result = layout(map);
  const { positions } = result;

  // Determinism — a second call is structurally identical.
  expect(layout(map)).toEqual(result);

  // Positive, bounded canvas; every box sits inside it.
  expect(result.bounds.w).toBeGreaterThan(0);
  expect(result.bounds.h).toBeGreaterThan(0);
  for (const box of positions.values()) {
    expect(box.x).toBeGreaterThanOrEqual(0);
    expect(box.y).toBeGreaterThanOrEqual(0);
    expect(box.x + box.w).toBeLessThanOrEqual(result.bounds.w);
    expect(box.y + box.h).toBeLessThanOrEqual(result.bounds.h);
  }

  // No two node boxes overlap.
  const entries = [...positions.entries()];
  for (let i = 0; i < entries.length; i += 1) {
    for (let j = i + 1; j < entries.length; j += 1) {
      expect(
        overlaps(entries[i][1], entries[j][1]),
        `${entries[i][0]} overlaps ${entries[j][0]}`,
      ).toBe(false);
    }
  }

  // Every edge is drawable left->right or intra-column (never rank-decreasing),
  // and its path starts/ends on its endpoints' box boundaries.
  const column = columnIndex(positions);
  for (const { edge, d } of result.edgePaths) {
    const from = boxFor(positions, edge.from);
    const to = boxFor(positions, edge.to);
    expect(
      (column.get(edge.to) ?? 0) >= (column.get(edge.from) ?? 0),
      `${edge.from} -> ${edge.to} decreases rank`,
    ).toBe(true);
    const { start, end } = endpoints(d);
    expect(onBoundary(start, from), `${edge.from} -> ${edge.to} start off box`).toBe(
      true,
    );
    expect(onBoundary(end, to), `${edge.from} -> ${edge.to} end off box`).toBe(true);
  }

  // Group bands: one per group, every member inside, columns consecutive,
  // members consecutive within each column.
  const bandOf = new Map(result.groupBands.map((band) => [band.group, band]));
  expect([...bandOf.keys()].sort()).toEqual(groupNames(map).sort());

  for (const group of groupNames(map)) {
    const band = bandOf.get(group);
    if (band === undefined) throw new Error(`no band for "${group}"`);
    const memberIds = map.graph.nodes
      .filter((node) => node.group === group)
      .map((node) => node.id);

    for (const id of memberIds) {
      const box = boxFor(positions, id);
      expect(box.x).toBeGreaterThanOrEqual(band.x);
      expect(box.y).toBeGreaterThanOrEqual(band.y);
      expect(box.x + box.w).toBeLessThanOrEqual(band.x + band.w);
      expect(box.y + box.h).toBeLessThanOrEqual(band.y + band.h);
    }

    // Occupied columns are a consecutive run.
    const cols = [...new Set(memberIds.map((id) => column.get(id) ?? 0))].sort(
      (p, q) => p - q,
    );
    expect(cols[cols.length - 1] - cols[0] + 1).toBe(cols.length);

    // Within each occupied column the members sit in consecutive slots.
    for (const c of cols) {
      const colX = xForColumn(positions, c);
      const ordered = [...positions.entries()]
        .filter(([, box]) => box.x === colX)
        .sort((p, q) => p[1].y - q[1].y)
        .map(([id]) => id);
      const slots = memberIds
        .filter((id) => (column.get(id) ?? 0) === c)
        .map((id) => ordered.indexOf(id))
        .sort((p, q) => p - q);
      expect(slots[slots.length - 1] - slots[0] + 1).toBe(slots.length);
    }
  }
}

/** The shared x of every node in a given column index. */
function xForColumn(positions: ReadonlyMap<string, NodeBox>, c: number): number {
  const xs = [...new Set([...positions.values()].map((box) => box.x))].sort(
    (p, q) => p - q,
  );
  return xs[c];
}

describe("layout — fixture", () => {
  it("holds every layout invariant", () => {
    assertLayoutProperties(FIXTURE);
  });

  it("ranks by longest path (the long edge doesn't shorten z's column)", () => {
    const column = columnIndex(layout(FIXTURE).positions);
    expect(column.get("a")).toBe(0);
    expect(column.get("b")).toBe(0);
    expect(column.get("z")).toBe(3);
    expect(column.get("g3")).toBe(2);
  });

  it("keeps the group's members contiguous even beside an ungrouped node", () => {
    const { positions } = layout(FIXTURE);
    const column = columnIndex(positions);
    const firstColMembers = ["g1", "g2"].filter((id) => (column.get(id) ?? 0) === 1);
    const ordered = [...positions.entries()]
      .filter(([id]) => (column.get(id) ?? 0) === 1)
      .sort((p, q) => p[1].y - q[1].y)
      .map(([id]) => id);
    const slots = firstColMembers.map((id) => ordered.indexOf(id)).sort((p, q) => p - q);
    expect(slots[slots.length - 1] - slots[0] + 1).toBe(slots.length);
  });

  it("throws on a cyclic graph", () => {
    expect(() => layout(CYCLE)).toThrow(/cycle/);
  });
});

describe("layout — real CODEBASE_MAP", () => {
  it("holds every layout invariant", () => {
    assertLayoutProperties(CODEBASE_MAP);
  });

  it("draws every edge (none dropped as dangling)", () => {
    const result = layout(CODEBASE_MAP);
    expect(result.edgePaths).toHaveLength(CODEBASE_MAP.graph.edges.length);
  });

  it("places every node exactly once", () => {
    const result = layout(CODEBASE_MAP);
    expect(result.positions.size).toBe(CODEBASE_MAP.graph.nodes.length);
  });
});
