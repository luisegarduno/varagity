import { describe, expect, it } from "vitest";

import {
  buildGraphView,
  DEFAULT_MAX_NODES,
  EDGE_COLOR,
  entityTypeColor,
  filterEntities,
  MAX_EXPORT_NODES,
  neighborhood,
  nodeSize,
  normalizeEntityType,
  parseMaxNodes,
  seedPosition,
  UNTYPED,
  type GraphExport,
  type GraphExportEdge,
  type GraphExportNode,
} from "@/lib/graph-view";

function node(
  id: string,
  overrides: Partial<GraphExportNode> = {},
): GraphExportNode {
  return {
    id,
    entity_type: null,
    description: null,
    degree: 0,
    ...overrides,
  };
}

function edge(
  source: string,
  target: string,
  overrides: Partial<GraphExportEdge> = {},
): GraphExportEdge {
  return {
    id: `${source}-${target}`,
    source,
    target,
    label: null,
    description: null,
    ...overrides,
  };
}

function exported(
  nodes: GraphExportNode[],
  edges: GraphExportEdge[] = [],
  truncated = false,
): GraphExport {
  return { nodes, edges, truncated };
}

describe("normalizeEntityType", () => {
  it("folds case and whitespace so one concept draws as one type", () => {
    expect(normalizeEntityType("Person")).toBe("person");
    expect(normalizeEntityType("  PERSON ")).toBe("person");
  });

  it("names the untyped rather than leaving a blank legend row", () => {
    expect(normalizeEntityType(null)).toBe(UNTYPED);
    expect(normalizeEntityType(undefined)).toBe(UNTYPED);
    expect(normalizeEntityType("   ")).toBe(UNTYPED);
  });
});

describe("entityTypeColor", () => {
  it("pins the types the extraction actually emits", () => {
    expect(entityTypeColor("person")).toBe("#4c8dff");
    expect(entityTypeColor("person")).not.toBe(entityTypeColor("event"));
  });

  it("is stable for a type it does not pin", () => {
    // ★ A hashed colour that shuffled per render would make the legend a lie.
    expect(entityTypeColor("spacecraft")).toBe(entityTypeColor("spacecraft"));
    expect(entityTypeColor("spacecraft")).toMatch(/^#[0-9a-f]{6}$/);
  });
});

describe("nodeSize", () => {
  it("grows with degree, bounded at both ends", () => {
    expect(nodeSize(0, 10)).toBeLessThan(nodeSize(5, 10));
    expect(nodeSize(5, 10)).toBeLessThan(nodeSize(10, 10));
    expect(nodeSize(10, 10)).toBeCloseTo(18);
  });

  it("draws an edgeless slice at the floor rather than dividing by zero", () => {
    expect(nodeSize(0, 0)).toBe(4);
  });
});

describe("seedPosition", () => {
  it("is deterministic, so the same graph draws the same picture", () => {
    expect(seedPosition(3, 8)).toEqual(seedPosition(3, 8));
  });

  it("spreads nodes apart — a shared origin never resolves under FA2", () => {
    const first = seedPosition(0, 4);
    const second = seedPosition(1, 4);
    expect(first.x === second.x && first.y === second.y).toBe(false);
  });

  it("survives an empty slice", () => {
    expect(seedPosition(0, 0)).toEqual({ x: 100, y: 0 });
  });
});

describe("parseMaxNodes", () => {
  it("defaults when the URL carries no override", () => {
    expect(parseMaxNodes(null)).toBe(DEFAULT_MAX_NODES);
    expect(parseMaxNodes("")).toBe(DEFAULT_MAX_NODES);
    expect(parseMaxNodes("  ")).toBe(DEFAULT_MAX_NODES);
  });

  it("honours the dev check's forced-low value", () => {
    expect(parseMaxNodes("10")).toBe(10);
  });

  it("clamps into the server's accepted range — never a 422 from a URL", () => {
    expect(parseMaxNodes("5000")).toBe(MAX_EXPORT_NODES);
    expect(parseMaxNodes("0")).toBe(1);
    expect(parseMaxNodes("-3")).toBe(1);
  });

  it("falls back on garbage rather than propagating NaN", () => {
    expect(parseMaxNodes("plenty")).toBe(DEFAULT_MAX_NODES);
  });
});

describe("buildGraphView", () => {
  it("maps nodes onto drawable attributes, coloured by type", () => {
    const model = buildGraphView(
      exported([
        node("Bob", { entity_type: "Person", description: "a friend", degree: 2 }),
        node("Keyboard", { entity_type: "technology", degree: 1 }),
      ]),
    );
    expect(model.nodes.map((item) => item.key)).toEqual(["Bob", "Keyboard"]);
    expect(model.nodes[0].entityType).toBe("person");
    expect(model.nodes[0].color).toBe(entityTypeColor("person"));
    expect(model.nodes[0].description).toBe("a friend");
    expect(model.nodes[0].size).toBeGreaterThan(model.nodes[1].size);
  });

  it("drops the edges graphology would throw on", () => {
    // ★ Any one of these blanks the whole canvas rather than one edge.
    const model = buildGraphView(
      exported(
        [node("Bob"), node("Keyboard")],
        [
          edge("Bob", "Keyboard"),
          edge("Bob", "Keyboard"), // duplicate key
          edge("Bob", "Ghost"), // endpoint not in the slice
          edge("Bob", "Bob"), // self-loop: draws nothing, says nothing
        ],
      ),
    );
    expect(model.edges.map((item) => item.key)).toEqual(["Bob-Keyboard"]);
    expect(model.edges[0].color).toBe(EDGE_COLOR);
  });

  it("drops duplicate and nameless nodes", () => {
    const model = buildGraphView(
      exported([node("Bob"), node("Bob"), node("")]),
    );
    expect(model.nodes).toHaveLength(1);
  });

  it("counts the legend by type, biggest group first", () => {
    const model = buildGraphView(
      exported([
        node("Bob", { entity_type: "person" }),
        node("Jane", { entity_type: "person" }),
        node("Keyboard", { entity_type: "technology" }),
        node("Thing"),
      ]),
    );
    expect(model.legend).toEqual([
      { entityType: "person", color: entityTypeColor("person"), count: 2 },
      // Ties break alphabetically, so the legend order never depends on
      // whatever order the engine happened to return the nodes in.
      {
        entityType: "technology",
        color: entityTypeColor("technology"),
        count: 1,
      },
      { entityType: UNTYPED, color: entityTypeColor(UNTYPED), count: 1 },
    ]);
  });

  it("passes truncation through — the view must say it drew a slice", () => {
    expect(buildGraphView(exported([node("Bob")], [], true)).truncated).toBe(
      true,
    );
    expect(buildGraphView(exported([])).truncated).toBe(false);
  });
});

describe("filterEntities", () => {
  const nodes = buildGraphView(
    exported([
      node("Bobbi's landlord"),
      node("Bob Nakamura"),
      node("Jane"),
      node("bobsled"),
    ]),
  ).nodes;

  it("is empty for an empty query — the canvas already shows everything", () => {
    expect(filterEntities(nodes, "   ")).toEqual([]);
  });

  it("ranks prefix hits above substring hits, case-insensitively", () => {
    expect(filterEntities(nodes, "bob").map((item) => item.label)).toEqual([
      "Bobbi's landlord",
      "Bob Nakamura",
      "bobsled",
    ]);
  });

  it("caps the result list", () => {
    expect(filterEntities(nodes, "b", 2)).toHaveLength(2);
  });
});

describe("neighborhood", () => {
  const edges = buildGraphView(
    exported(
      [node("Bob"), node("Keyboard"), node("Jane"), node("Alone")],
      [edge("Bob", "Keyboard"), edge("Jane", "Bob")],
    ),
  ).edges;

  it("lights the focused node and everything it touches", () => {
    expect(neighborhood(edges, "Bob")).toEqual(
      new Set(["Bob", "Keyboard", "Jane"]),
    );
  });

  it("lights only itself when nothing connects to it", () => {
    expect(neighborhood(edges, "Alone")).toEqual(new Set(["Alone"]));
  });

  it("lights nothing when nothing is focused", () => {
    expect(neighborhood(edges, null).size).toBe(0);
  });
});
