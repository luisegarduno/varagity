import { existsSync, readFileSync, statSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

import { CODEBASE_MAP } from "@/lib/codebase-map.data";
import { validateMap, type CodebaseMap } from "@/lib/codebase-map";

// web/lib/__tests__ → repo root (the parents[2] idiom from
// tests/unit/test_openapi_snapshot.py:15, resolved for this file's depth).
const REPO_ROOT = path.resolve(import.meta.dirname, "../../..");

/** Split "path" | "path:line" into its parts. */
function splitRef(ref: string): { file: string; line?: number } {
  const match = /^(.*):(\d+)$/.exec(ref);
  if (match) return { file: match[1], line: Number(match[2]) };
  return { file: ref };
}

function isDirectory(abs: string): boolean {
  try {
    return statSync(abs).isDirectory();
  } catch {
    return false;
  }
}

/** A mutable deep copy, so a negative test can break one thing at a time. */
function clone(): CodebaseMap {
  return JSON.parse(JSON.stringify(CODEBASE_MAP)) as CodebaseMap;
}

describe("CODEBASE_MAP shape", () => {
  it("validateMap accepts the checked-in map (invariants 1–6, 8)", () => {
    expect(validateMap(CODEBASE_MAP)).toEqual([]);
  });

  it("matches the corrected Phase 1 totals", () => {
    const { nodes, edges } = CODEBASE_MAP.graph;
    expect(nodes).toHaveLength(41);
    expect(edges).toHaveLength(59);
    expect(CODEBASE_MAP.topModels).toHaveLength(3);
    expect(CODEBASE_MAP.topTools).toHaveLength(8);
    expect(CODEBASE_MAP.topIntegrations).toHaveLength(7);

    const groups = new Map<string, number>();
    for (const node of nodes) {
      if (node.group) groups.set(node.group, (groups.get(node.group) ?? 0) + 1);
    }
    expect([...groups.entries()].sort()).toEqual([
      ["Chat engines", 3],
      ["Ingestion", 5],
      ["Web app", 4],
    ]);
  });

  it("pins web-map to its page and keeps store-corpus sourceRef-free", () => {
    const byId = new Map(CODEBASE_MAP.graph.nodes.map((node) => [node.id, node]));
    // Phase 3 created web/app/map/page.tsx, so the map now pins itself to it
    // (guard and artifact land together); store-corpus stays sourceRef-free
    // because DOCS_PATH is gitignored and could never resolve on a CI checkout.
    expect(byId.get("web-map")?.sourceRef).toBe("web/app/map/page.tsx");
    expect(byId.get("store-corpus")?.sourceRef).toBeUndefined();
  });

  // The test_dashboards.py:111 idiom: assert a floor of real data so the drift
  // guard below cannot pass vacuously on an empty or gutted map.
  it("carries enough data that the drift guard can't pass vacuously", () => {
    const { nodes, edges } = CODEBASE_MAP.graph;
    expect(nodes.length).toBeGreaterThanOrEqual(40);
    expect(edges.length).toBeGreaterThanOrEqual(55);
    const withSourceRef = nodes.filter((node) => node.sourceRef).length;
    expect(withSourceRef).toBeGreaterThanOrEqual(25);
  });
});

describe("sourceRef drift guard (invariant 7)", () => {
  it("resolves every sourceRef path against the repo root", () => {
    for (const node of CODEBASE_MAP.graph.nodes) {
      if (!node.sourceRef) continue;
      const { file, line } = splitRef(node.sourceRef);
      const abs = path.join(REPO_ROOT, file);
      expect(existsSync(abs), `${node.id}: ${file} is missing`).toBe(true);
      if (line !== undefined && !isDirectory(abs)) {
        const count = readFileSync(abs, "utf8").split("\n").length;
        expect(
          count,
          `${node.id}: ${file} has fewer than ${line} lines`,
        ).toBeGreaterThanOrEqual(line);
      }
    }
  });
});

describe("validateMap catches violations", () => {
  it("flags a duplicate node id", () => {
    const map = clone();
    map.graph.nodes.push({ ...map.graph.nodes[0] });
    expect(validateMap(map).some((e) => e.includes("duplicate node id"))).toBe(true);
  });

  it("flags a non-kebab-case node id", () => {
    const map = clone();
    map.graph.nodes.push({ id: "Not_Kebab", label: "x", kind: "tool" });
    expect(validateMap(map).some((e) => e.includes("not kebab-case"))).toBe(true);
  });

  it("flags an edge with an unknown 'from' node", () => {
    const map = clone();
    map.graph.edges.push({ from: "ghost", to: "web-chat" });
    expect(validateMap(map).some((e) => e.includes('unknown "from" node'))).toBe(true);
  });

  it("flags an edge with an unknown 'to' node", () => {
    const map = clone();
    map.graph.edges.push({ from: "web-chat", to: "ghost" });
    expect(validateMap(map).some((e) => e.includes('unknown "to" node'))).toBe(true);
  });

  it("flags a self-edge", () => {
    const map = clone();
    map.graph.edges.push({ from: "web-chat", to: "web-chat" });
    expect(validateMap(map).some((e) => e.includes("self-edge"))).toBe(true);
  });

  it("flags too many nodes", () => {
    const map = clone();
    for (let i = 0; i < 61; i += 1) {
      map.graph.nodes.push({ id: `extra-${i}`, label: "x", kind: "tool" });
    }
    expect(validateMap(map).some((e) => e.includes("too many nodes"))).toBe(true);
  });

  it("flags too many edges", () => {
    const map = clone();
    for (let i = 0; i < 121; i += 1) {
      map.graph.edges.push({ from: "web-chat", to: "api-chat" });
    }
    expect(validateMap(map).some((e) => e.includes("too many edges"))).toBe(true);
  });

  it("flags an oversized top* list", () => {
    const model = clone();
    model.topModels = ["llm-chat", "model-embed", "model-rerank", "llm-chat"];
    expect(validateMap(model).some((e) => e.includes("too many topModels"))).toBe(true);

    const tools = clone();
    tools.topTools = Array<string>(11).fill("retriever-hybrid");
    expect(validateMap(tools).some((e) => e.includes("too many topTools"))).toBe(true);

    const integrations = clone();
    integrations.topIntegrations = Array<string>(11).fill("store-postgres");
    expect(
      validateMap(integrations).some((e) => e.includes("too many topIntegrations")),
    ).toBe(true);
  });

  it("flags over-length fields", () => {
    const label = clone();
    label.graph.nodes[0].label = "x".repeat(29);
    expect(validateMap(label).some((e) => e.includes("label > 28"))).toBe(true);

    const sub = clone();
    sub.graph.nodes[0].sub = "x".repeat(41);
    expect(validateMap(sub).some((e) => e.includes("sub > 40"))).toBe(true);

    const detail = clone();
    detail.graph.nodes[0].detail = "x".repeat(201);
    expect(validateMap(detail).some((e) => e.includes("detail > 200"))).toBe(true);

    const sourceRef = clone();
    sourceRef.graph.nodes[0].sourceRef = "x".repeat(121);
    expect(validateMap(sourceRef).some((e) => e.includes("sourceRef > 120"))).toBe(true);

    const group = clone();
    group.graph.nodes[0].group = "x".repeat(25);
    expect(validateMap(group).some((e) => e.includes("group > 24"))).toBe(true);

    const edgeLabel = clone();
    edgeLabel.graph.edges[0].label = "x".repeat(25);
    expect(validateMap(edgeLabel).some((e) => e.includes("label > 24"))).toBe(true);
  });

  it("flags a top* id that is not a node", () => {
    const map = clone();
    map.topModels[0] = "ghost-model";
    expect(
      validateMap(map).some((e) => e.includes("topModels references unknown node")),
    ).toBe(true);
  });

  it("flags too many distinct groups", () => {
    const map = clone();
    for (let i = 0; i < 3; i += 1) {
      map.graph.nodes.push({ id: `extra-${i}`, label: "x", kind: "tool", group: "Extra" });
    }
    expect(
      validateMap(map).some((e) => e.includes("too many distinct groups")),
    ).toBe(true);
  });

  it("flags a group outside the 3–6 member range", () => {
    const map = clone();
    for (let i = 0; i < 4; i += 1) {
      map.graph.nodes.push({ id: `wa-${i}`, label: "x", kind: "tool", group: "Web app" });
    }
    expect(validateMap(map).some((e) => e.includes('group "Web app" holds 8'))).toBe(true);
  });

  it("flags a domain that carries a scheme or path", () => {
    const map = clone();
    map.graph.nodes[0].domain = "https://nextjs.org";
    expect(
      validateMap(map).some((e) => e.includes("must have no scheme or path")),
    ).toBe(true);
  });

  it("flags a cycle (not a DAG)", () => {
    const map = clone();
    // api-chat -> flow-query already exists; close the loop.
    map.graph.edges.push({ from: "flow-query", to: "api-chat" });
    expect(validateMap(map).some((e) => e.includes("not a DAG"))).toBe(true);
  });
});
