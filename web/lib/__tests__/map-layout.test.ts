import { describe, expect, it } from "vitest";

import { CODEBASE_MAP } from "@/lib/codebase-map.data";
import { precomputedLayout } from "@/lib/codebase-map.layout";
import type { CodebaseMap } from "@/lib/codebase-map";
import {
  arrowHead,
  condense,
  edgePath,
  labelAnchor,
  layoutGraph,
  polylineLength,
  serializeLayout,
  type MapLayout,
  type NodeBox,
} from "@/lib/map-layout";

/** A layout-only fixture: a group, a chip-folded model, an ungrouped chain,
 * and a long edge. It need not satisfy validateMap — the layout trusts its
 * input. */
const FIXTURE: CodebaseMap = {
  project: { name: "Fixture", date: "2026-01-01", tagline: "layout fixture" },
  topModels: [],
  topTools: [],
  topIntegrations: [],
  graph: {
    nodes: [
      { id: "a", label: "A", kind: "entry", sub: "source" },
      { id: "b", label: "B", kind: "entry" },
      { id: "g1", label: "G1", kind: "service", group: "Group" },
      { id: "g2", label: "G2", kind: "service", group: "Group" },
      { id: "g3", label: "G3", kind: "agent", group: "Group" },
      { id: "mid", label: "MID", kind: "service" },
      { id: "z", label: "Z", kind: "store" },
      { id: "m", label: "M", kind: "model", domain: "example.com" },
    ],
    edges: [
      { from: "a", to: "g1", label: "starts" },
      { from: "a", to: "mid" },
      { from: "a", to: "z" }, // long edge across every layer
      { from: "b", to: "g2" },
      { from: "b", to: "mid" },
      { from: "g1", to: "g3", label: "hands off" },
      { from: "g2", to: "g3" },
      { from: "mid", to: "m" }, // chip-folded
      { from: "g3", to: "m" }, // chip-folded
      { from: "g3", to: "z" },
      { from: "mid", to: "z" },
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

/** The invariants that must hold for any graph the layout accepts. */
async function assertLayoutProperties(map: CodebaseMap): Promise<MapLayout> {
  const result = await layoutGraph(map);
  const { positions } = result;

  // Determinism — a second run is structurally identical (ELK layered is
  // deterministic; same input → same coordinates).
  expect(JSON.parse(JSON.stringify(await layoutGraph(map)))).toEqual(
    JSON.parse(JSON.stringify(result)),
  );

  // Exactly the condensed (renderable) nodes are placed.
  const condensed = condense(map);
  expect([...positions.keys()].sort()).toEqual(
    condensed.nodes.map((node) => node.id).sort(),
  );
  expect(result.foldedEdges).toEqual(condensed.edges);

  // Every condensed edge is represented exactly once across the rendered
  // edges' `orig` lists (cross-group edges merge; none may vanish).
  const seen = new Map<number, number>();
  for (const edge of result.edges) {
    for (const index of edge.orig) seen.set(index, (seen.get(index) ?? 0) + 1);
  }
  expect([...seen.values()].every((count) => count === 1)).toBe(true);
  expect(seen.size).toBe(condensed.edges.length);

  // Rendered endpoints resolve to a placed node or a group band.
  const bandIds = new Set(result.groupBands.map((band) => band.id));
  for (const edge of result.edges) {
    expect(positions.has(edge.from) || bandIds.has(edge.from)).toBe(true);
    expect(positions.has(edge.to) || bandIds.has(edge.to)).toBe(true);
    expect(edge.points.length).toBeGreaterThanOrEqual(2);
    for (const p of edge.points) {
      expect(Number.isFinite(p.x)).toBe(true);
      expect(Number.isFinite(p.y)).toBe(true);
    }
    if (edge.labelPos) {
      expect(Number.isFinite(edge.labelPos.x)).toBe(true);
      expect(Number.isFinite(edge.labelPos.y)).toBe(true);
    }
    expect(polylineLength(edge.points)).toBeGreaterThan(0);
    expect(edgePath(edge.points).startsWith("M ")).toBe(true);
    expect(arrowHead(edge.points).startsWith("M ")).toBe(true);
    expect(labelAnchor(edge.points)).not.toBeNull();
  }

  // Positive, bounded canvas; every box sits inside it; nothing is NaN.
  expect(result.bounds.w).toBeGreaterThan(0);
  expect(result.bounds.h).toBeGreaterThan(0);
  for (const box of positions.values()) {
    for (const v of [box.x, box.y, box.w, box.h]) {
      expect(Number.isFinite(v)).toBe(true);
    }
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

  // Group containers: one per group, every member strictly inside, and no
  // foreign card intruding.
  const groupNames = [
    ...new Set(map.graph.nodes.flatMap((node) => (node.group ? [node.group] : []))),
  ];
  expect(result.groupBands.map((band) => band.group).sort()).toEqual(
    [...groupNames].sort(),
  );
  for (const band of result.groupBands) {
    const memberIds = map.graph.nodes
      .filter((node) => node.group === band.group)
      .map((node) => node.id);
    for (const id of memberIds) {
      const box = boxFor(positions, id);
      expect(box.x).toBeGreaterThan(band.x);
      expect(box.y).toBeGreaterThan(band.y);
      expect(box.x + box.w).toBeLessThan(band.x + band.w);
      expect(box.y + box.h).toBeLessThan(band.y + band.h);
    }
    for (const [id, box] of positions) {
      if (memberIds.includes(id)) continue;
      expect(overlaps(box, band), `${id} intrudes into "${band.group}"`).toBe(false);
    }
  }

  return result;
}

describe("condense", () => {
  it("folds models into chips on their callers", () => {
    const { nodes, edges, chips } = condense(FIXTURE);
    expect(nodes.map((node) => node.id)).not.toContain("m");
    expect(edges.some((edge) => edge.to === "m")).toBe(false);
    expect(chips.get("mid")).toEqual([{ label: "M", domain: "example.com" }]);
    expect(chips.get("g3")).toEqual([{ label: "M", domain: "example.com" }]);
  });

  it("folds the real map's three models into the expected cards", () => {
    const { nodes, chips } = condense(CODEBASE_MAP);
    expect(nodes).toHaveLength(23);
    expect(nodes.some((node) => node.kind === "model")).toBe(false);
    expect(chips.get("retrievers")?.map((chip) => chip.label)).toEqual([
      "multilingual-e5-large",
      "bge-reranker-v2-m3",
    ]);
    expect(chips.get("ingest")?.map((chip) => chip.label)).toEqual([
      "multilingual-e5-large",
    ]);
    for (const id of ["condenser", "answerer", "contextualizer"]) {
      expect(chips.get(id)?.map((chip) => chip.label)).toEqual([
        "Qwythos-9B (llama.cpp)",
      ]);
    }
  });
});

describe("layoutGraph — fixture", () => {
  it("holds every layout invariant", async () => {
    await assertLayoutProperties(FIXTURE);
  });

  it("sizes chip-carrying cards taller than plain ones", async () => {
    const { positions } = await layoutGraph(FIXTURE);
    expect(boxFor(positions, "mid").h).toBeGreaterThan(boxFor(positions, "a").h);
  });
});

describe("layoutGraph — real CODEBASE_MAP", () => {
  it("holds every layout invariant", async () => {
    await assertLayoutProperties(CODEBASE_MAP);
  });

  it("draws the three group containers", async () => {
    const result = await layoutGraph(CODEBASE_MAP);
    expect(result.groupBands.map((band) => band.group).sort()).toEqual([
      "Ingestion",
      "Observability",
      "Query path",
    ]);
  });

  it("merges same-pair cross-group edges and keeps their origins", async () => {
    const result = await layoutGraph(CODEBASE_MAP);
    // api → Query path carries both api→condenser and api→qflow.
    const merged = result.edges.find(
      (edge) => edge.from === "api" && edge.orig.length > 1,
    );
    expect(merged).toBeDefined();
  });
});

describe("precomputed layout snapshot", () => {
  // The page renders the checked-in snapshot (instant hydration, no elkjs in
  // the browser) — both guards below fail CI when the data changes without a
  // regeneration. Regenerate with: bun run test -u

  it("codebase-map.layout.json matches a live ELK run", async () => {
    const live = serializeLayout(await layoutGraph(CODEBASE_MAP));
    await expect(`${JSON.stringify(live, null, 2)}\n`).toMatchFileSnapshot(
      "../codebase-map.layout.json",
    );
  });

  it("the loader rehydrates exactly what ELK produces", async () => {
    const live = JSON.parse(
      JSON.stringify(serializeLayout(await layoutGraph(CODEBASE_MAP))),
    );
    expect(serializeLayout(precomputedLayout())).toEqual(live);
  });
});
