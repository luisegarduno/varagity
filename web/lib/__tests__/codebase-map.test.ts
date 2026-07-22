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

  it("matches the curated totals", () => {
    const { nodes, edges } = CODEBASE_MAP.graph;
    expect(nodes).toHaveLength(26);
    expect(edges).toHaveLength(38);
    expect(CODEBASE_MAP.topModels).toHaveLength(3);
    expect(CODEBASE_MAP.topTools).toHaveLength(6);
    expect(CODEBASE_MAP.topIntegrations).toHaveLength(10);

    const groups = new Map<string, number>();
    for (const node of nodes) {
      if (node.group) groups.set(node.group, (groups.get(node.group) ?? 0) + 1);
    }
    expect([...groups.entries()].sort()).toEqual([
      ["Ingestion", 5],
      ["Observability", 3],
      ["Query path", 4],
    ]);
  });

  it("pins the map's own page and keeps docsdir sourceRef-free", () => {
    const byId = new Map(CODEBASE_MAP.graph.nodes.map((node) => [node.id, node]));
    // The map's update rule applied to itself (ADR-015): the web entry pins
    // /map's page, so deleting the map route fails CI. The chat page the web
    // node previously pinned keeps an existence guard here instead.
    expect(byId.get("web")?.sourceRef).toBe("web/app/map/page.tsx");
    expect(existsSync(path.join(REPO_ROOT, "web/app/page.tsx"))).toBe(true);
    // docs/ is gitignored, so its node could never resolve on a CI checkout.
    expect(byId.get("docsdir")?.sourceRef).toBeUndefined();
  });

  // The test_dashboards.py:111 idiom: assert a floor of real data so the drift
  // guard below cannot pass vacuously on an empty or gutted map.
  it("carries enough data that the drift guard can't pass vacuously", () => {
    const { nodes, edges } = CODEBASE_MAP.graph;
    expect(nodes.length).toBeGreaterThanOrEqual(20);
    expect(edges.length).toBeGreaterThanOrEqual(30);
    const withSourceRef = nodes.filter((node) => node.sourceRef).length;
    expect(withSourceRef).toBeGreaterThanOrEqual(12);
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
    map.graph.edges.push({ from: "ghost", to: "web" });
    expect(validateMap(map).some((e) => e.includes('unknown "from" node'))).toBe(true);
  });

  it("flags an edge with an unknown 'to' node", () => {
    const map = clone();
    map.graph.edges.push({ from: "web", to: "ghost" });
    expect(validateMap(map).some((e) => e.includes('unknown "to" node'))).toBe(true);
  });

  it("flags a self-edge", () => {
    const map = clone();
    map.graph.edges.push({ from: "web", to: "web" });
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
      map.graph.edges.push({ from: "web", to: "api" });
    }
    expect(validateMap(map).some((e) => e.includes("too many edges"))).toBe(true);
  });

  it("flags an oversized top* list", () => {
    const item = { id: "extra", label: "Extra" };

    const model = clone();
    model.topModels = [item, item, item, item];
    expect(validateMap(model).some((e) => e.includes("too many topModels"))).toBe(true);

    const tools = clone();
    tools.topTools = Array(11).fill(item);
    expect(validateMap(tools).some((e) => e.includes("too many topTools"))).toBe(true);

    const integrations = clone();
    integrations.topIntegrations = Array(11).fill(item);
    expect(
      validateMap(integrations).some((e) => e.includes("too many topIntegrations")),
    ).toBe(true);
  });

  it("flags malformed top* rows", () => {
    const kebab = clone();
    kebab.topModels[0] = { id: "Not_Kebab", label: "x" };
    expect(
      validateMap(kebab).some((e) => e.includes("topModels id is not kebab-case")),
    ).toBe(true);

    const dupe = clone();
    dupe.topIntegrations[1] = { ...dupe.topIntegrations[0] };
    expect(
      validateMap(dupe).some((e) => e.includes("topIntegrations has a duplicate id")),
    ).toBe(true);

    const label = clone();
    label.topModels[0] = { id: "extra", label: "x".repeat(41) };
    expect(
      validateMap(label).some((e) => e.includes("label must be 1–40 chars")),
    ).toBe(true);

    const domain = clone();
    domain.topIntegrations[0] = {
      id: "extra",
      label: "Extra",
      domain: "https://nextjs.org",
    };
    expect(
      validateMap(domain).some((e) => e.includes("must have no scheme or path")),
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
      map.graph.nodes.push({
        id: `obs-${i}`,
        label: "x",
        kind: "tool",
        group: "Observability",
      });
    }
    expect(
      validateMap(map).some((e) => e.includes('group "Observability" holds 7')),
    ).toBe(true);
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
    // api -> qflow already exists; close the loop.
    map.graph.edges.push({ from: "qflow", to: "api" });
    expect(validateMap(map).some((e) => e.includes("not a DAG"))).toBe(true);
  });
});
