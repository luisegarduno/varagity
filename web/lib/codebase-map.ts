/**
 * Schema + invariant validator for the curated codebase map
 * (thoughts/shared/tasks/codebase_map/spec_codebase_map.md §4).
 *
 * The data lives in {@link ./codebase-map.data.ts}; {@link validateMap} is
 * exercised only by tests — the app trusts the checked-in data. Authoring the
 * data as `… satisfies CodebaseMap` (not JSON) is what makes the `kind` /
 * `EdgeKind` unions type-check at build time; a JSON literal would widen to
 * `string` and silently accept a typo (microsoft/TypeScript#26552, ADR-015).
 */

/** The shape a node plays in the system; drives its glyph in §5.8. */
export type NodeKind =
  | "entry"
  | "agent"
  | "model"
  | "tool"
  | "service"
  | "store"
  | "external";

/** The relationship an edge asserts; revealed when a flow is traced. */
export type EdgeKind = "calls" | "reads" | "writes" | "triggers";

/** One vertex of the map. */
export interface MapNode {
  /** Unique, kebab-case identifier. */
  id: string;
  /** Display name, <= 28 chars. */
  label: string;
  kind: NodeKind;
  /** One-line qualifier under the label, <= 40 chars. */
  sub?: string;
  /** Labeled band drawn behind co-members, <= 24 chars. */
  group?: string;
  /** Shown on click, <= 200 chars. */
  detail?: string;
  /** Repo-relative "path" or "path:line", <= 120 chars; drift-guarded in tests. */
  sourceRef?: string;
  /** Favicon domain, no scheme and no path; omit for internal nodes. */
  domain?: string;
}

/** One directed edge; `from`/`to` reference node ids. */
export interface MapEdge {
  from: string;
  to: string;
  kind?: EdgeKind;
  /** Always-visible caption — the interesting sentence, <= 24 chars. */
  label?: string;
}

/** The whole curated graph plus its "top" callouts. */
export interface CodebaseMap {
  project: { name: string; date: string; summary: string };
  /** Node ids, <= 3. */
  topModels: string[];
  /** Node ids, <= 10. */
  topTools: string[];
  /** Node ids, <= 10. */
  topIntegrations: string[];
  /** <= 60 nodes, <= 120 edges. */
  graph: { nodes: MapNode[]; edges: MapEdge[] };
}

const KEBAB_CASE = /^[a-z0-9]+(-[a-z0-9]+)*$/;

/**
 * Validate a {@link CodebaseMap} against the spec §4.2 invariants (corrected
 * and extended per the implementation plan).
 *
 * Enforced here (invariant 7, the `sourceRef`-exists drift guard, needs
 * `node:fs` and so lives in the test):
 *
 * 1. Node ids are unique and kebab-case; every edge endpoint resolves; no
 *    self-edges.
 * 2. Caps: <= 60 nodes, <= 120 edges, <= 3 topModels, <= 10 topTools,
 *    <= 10 topIntegrations.
 * 3. Lengths: label <= 28, sub <= 40, edge label <= 24, group <= 24,
 *    detail <= 200, sourceRef <= 120.
 * 4. Every `top*` id exists in `graph.nodes`.
 * 5. <= 3 distinct groups, each holding 3–6 nodes.
 * 6. `domain` carries no scheme and no path.
 * 8. The graph is a DAG (Kahn's algorithm) — the layout's longest-path rank
 *    step depends on it, so it is an invariant, not an assumption.
 *
 * @param map - The map to validate.
 * @returns Human-readable violations; an empty array means the map is valid.
 */
export function validateMap(map: CodebaseMap): string[] {
  const errors: string[] = [];
  const { nodes, edges } = map.graph;

  // Invariant 1 — ids unique + kebab-case.
  const ids = new Set<string>();
  for (const node of nodes) {
    if (!KEBAB_CASE.test(node.id)) {
      errors.push(`node id is not kebab-case: "${node.id}"`);
    }
    if (ids.has(node.id)) {
      errors.push(`duplicate node id: "${node.id}"`);
    }
    ids.add(node.id);
  }

  // Invariant 1 — edges resolve, no self-edges.
  for (const edge of edges) {
    if (!ids.has(edge.from)) {
      errors.push(`edge references unknown "from" node: "${edge.from}"`);
    }
    if (!ids.has(edge.to)) {
      errors.push(`edge references unknown "to" node: "${edge.to}"`);
    }
    if (edge.from === edge.to) {
      errors.push(`self-edge on node: "${edge.from}"`);
    }
  }

  // Invariant 2 — caps.
  if (nodes.length > 60) errors.push(`too many nodes: ${nodes.length} > 60`);
  if (edges.length > 120) errors.push(`too many edges: ${edges.length} > 120`);
  if (map.topModels.length > 3) {
    errors.push(`too many topModels: ${map.topModels.length} > 3`);
  }
  if (map.topTools.length > 10) {
    errors.push(`too many topTools: ${map.topTools.length} > 10`);
  }
  if (map.topIntegrations.length > 10) {
    errors.push(`too many topIntegrations: ${map.topIntegrations.length} > 10`);
  }

  // Invariant 3 — length limits.
  for (const node of nodes) {
    if (node.label.length > 28) {
      errors.push(`${node.id}: label > 28 chars (${node.label.length})`);
    }
    if (node.sub !== undefined && node.sub.length > 40) {
      errors.push(`${node.id}: sub > 40 chars (${node.sub.length})`);
    }
    if (node.group !== undefined && node.group.length > 24) {
      errors.push(`${node.id}: group > 24 chars (${node.group.length})`);
    }
    if (node.detail !== undefined && node.detail.length > 200) {
      errors.push(`${node.id}: detail > 200 chars (${node.detail.length})`);
    }
    if (node.sourceRef !== undefined && node.sourceRef.length > 120) {
      errors.push(`${node.id}: sourceRef > 120 chars (${node.sourceRef.length})`);
    }
  }
  for (const edge of edges) {
    if (edge.label !== undefined && edge.label.length > 24) {
      errors.push(
        `edge ${edge.from} -> ${edge.to}: label > 24 chars (${edge.label.length})`,
      );
    }
  }

  // Invariant 4 — top* ids exist.
  const checkTop = (name: string, list: readonly string[]): void => {
    for (const id of list) {
      if (!ids.has(id)) {
        errors.push(`${name} references unknown node: "${id}"`);
      }
    }
  };
  checkTop("topModels", map.topModels);
  checkTop("topTools", map.topTools);
  checkTop("topIntegrations", map.topIntegrations);

  // Invariant 5 — <= 3 groups, each 3–6 nodes.
  const groupCounts = new Map<string, number>();
  for (const node of nodes) {
    if (node.group !== undefined) {
      groupCounts.set(node.group, (groupCounts.get(node.group) ?? 0) + 1);
    }
  }
  if (groupCounts.size > 3) {
    errors.push(`too many distinct groups: ${groupCounts.size} > 3`);
  }
  for (const [group, count] of groupCounts) {
    if (count < 3 || count > 6) {
      errors.push(`group "${group}" holds ${count} nodes (must be 3–6)`);
    }
  }

  // Invariant 6 — domain has no scheme and no path.
  for (const node of nodes) {
    if (
      node.domain !== undefined &&
      (node.domain.includes("/") || node.domain.includes(":"))
    ) {
      errors.push(`${node.id}: domain "${node.domain}" must have no scheme or path`);
    }
  }

  // Invariant 8 — the graph is a DAG (Kahn's algorithm over resolvable edges).
  const indegree = new Map<string, number>();
  const adjacency = new Map<string, string[]>();
  for (const id of ids) indegree.set(id, 0);
  for (const edge of edges) {
    if (!ids.has(edge.from) || !ids.has(edge.to) || edge.from === edge.to) {
      continue;
    }
    indegree.set(edge.to, (indegree.get(edge.to) ?? 0) + 1);
    const out = adjacency.get(edge.from);
    if (out) out.push(edge.to);
    else adjacency.set(edge.from, [edge.to]);
  }
  const queue: string[] = [];
  for (const id of ids) {
    if ((indegree.get(id) ?? 0) === 0) queue.push(id);
  }
  let visited = 0;
  for (let head = 0; head < queue.length; head += 1) {
    const id = queue[head];
    visited += 1;
    for (const next of adjacency.get(id) ?? []) {
      const remaining = (indegree.get(next) ?? 0) - 1;
      indegree.set(next, remaining);
      if (remaining === 0) queue.push(next);
    }
  }
  if (visited !== ids.size) {
    errors.push(`graph is not a DAG: ${ids.size - visited} node(s) form a cycle`);
  }

  return errors;
}
