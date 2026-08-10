/**
 * Pure transforms behind the graph view (spec_graphrag §4.4).
 *
 * The export endpoint hands back entities and relations; sigma needs
 * positioned, coloured, sized node/edge *attributes*. Everything in between
 * is here — deliberately free of `graphology`, `sigma`, and React, so the
 * rules that decide what the picture means are unit-testable without a
 * WebGL context.
 *
 * Two honesty rules shape the module:
 *
 * * **Colour is entity type, not community.** LightRAG builds no community
 *   tier at all (ADR-017), so grouping by type is the real structure the
 *   engine extracted rather than a clustering invented in the browser. The
 *   legend names the types it drew for exactly that reason.
 * * **Seed positions are deterministic.** The layout runs from a fixed
 *   circular seed, so the same graph draws the same picture on every reload
 *   — a randomized seed would make "did the graph change?" unanswerable by
 *   eye. It also means the layout is settled before the first paint, so
 *   there is no animation for `prefers-reduced-motion` to suppress.
 */
import type { components } from "@/lib/types";

type Schemas = components["schemas"];

export type GraphExport = Schemas["GraphExportOut"];
export type GraphExportNode = Schemas["GraphExportNodeOut"];
export type GraphExportEdge = Schemas["GraphExportEdgeOut"];

/** One node, ready to hand to `graph.addNode(key, attributes)`. */
export interface GraphViewNode {
  key: string;
  label: string;
  x: number;
  y: number;
  size: number;
  color: string;
  /** Normalized entity type — the legend key and the colour channel. */
  entityType: string;
  description: string | null;
  degree: number;
}

/** One edge, ready to hand to `graph.addEdgeWithKey(key, …)`. */
export interface GraphViewEdge {
  key: string;
  source: string;
  target: string;
  /** `""` when the engine labelled the relation with nothing. */
  label: string;
  size: number;
  color: string;
}

/** One legend row: a drawn entity type, its colour, and how many wear it. */
export interface GraphViewLegendEntry {
  entityType: string;
  color: string;
  count: number;
}

/** Everything the canvas draws from one export. */
export interface GraphViewModel {
  nodes: GraphViewNode[];
  edges: GraphViewEdge[];
  legend: GraphViewLegendEntry[];
  truncated: boolean;
}

/** What an untyped node is grouped and labelled as. */
export const UNTYPED = "untyped";

/**
 * How many nodes the view asks for by default — the server's full ceiling:
 * the real archive's graph outgrew the old 1000-node slice (the 347-entity
 * eval-corpus measurement it was sized from), so the view now draws the
 * fullest slice the contract allows and `truncated` tells the truth when
 * even that bites.
 */
export const DEFAULT_MAX_NODES = 5000;

/** The server's export ceiling — above it the API answers `422`. */
export const MAX_EXPORT_NODES = 5000;

/**
 * Parse the `?max_nodes=` URL override.
 *
 * A dev affordance, not a product control: forcing the cap low (e.g.
 * `/graph?max_nodes=10`) is how the truncation notice is checked against a
 * real graph. Clamped into the server's accepted range so a hand-typed
 * value can never turn into a `422`; anything unparseable falls back to the
 * default.
 */
export function parseMaxNodes(raw: string | null): number {
  if (raw === null || raw.trim() === "") return DEFAULT_MAX_NODES;
  const parsed = Number.parseInt(raw, 10);
  if (Number.isNaN(parsed)) return DEFAULT_MAX_NODES;
  return Math.min(Math.max(parsed, 1), MAX_EXPORT_NODES);
}

/**
 * The node palette — literal hex, because sigma renders in WebGL and parses
 * only hex / `rgb()` / named colours (it cannot read a CSS custom property).
 * Mid-lightness, mid-chroma hues so the same swatch reads on both themes.
 * The first entries are pinned to the types LightRAG's extraction prompt
 * actually emits, so a person is the same blue in every graph; anything else
 * is hashed into the same list, deterministically.
 */
const PALETTE: Record<string, string> = {
  person: "#4c8dff",
  organization: "#f2994a",
  location: "#27ae8f",
  geo: "#27ae8f",
  event: "#c964dd",
  category: "#8a92a6",
  technology: "#e05c8a",
  product: "#e0b13c",
  [UNTYPED]: "#8a92a6",
};

/** The fallback ring, for types the palette does not pin by name. */
const FALLBACK_COLORS = [
  "#4c8dff",
  "#f2994a",
  "#27ae8f",
  "#c964dd",
  "#e05c8a",
  "#e0b13c",
  "#5bc0de",
  "#9b8cf5",
];

/** Edge colour: a neutral grey that recedes behind the nodes on both themes. */
export const EDGE_COLOR = "#9aa1ad";

/** Node radius bounds, in sigma units. */
const MIN_NODE_SIZE = 4;
const MAX_NODE_SIZE = 18;

/** How far the deterministic seed ring is from the origin. */
const SEED_RADIUS = 100;

/**
 * Normalize an engine entity type into a legend/palette key.
 *
 * The engine emits types in whatever case the extraction produced
 * (`Person`, `PERSON`, `person`), which would otherwise draw one concept in
 * three colours under three legend rows.
 */
export function normalizeEntityType(
  entityType: string | null | undefined,
): string {
  const trimmed = (entityType ?? "").trim().toLowerCase();
  return trimmed === "" ? UNTYPED : trimmed;
}

/**
 * The colour one (normalized) entity type draws in.
 *
 * Pinned types keep a fixed hue; anything else hashes into the fallback
 * ring, so an unexpected type is still stable across reloads instead of
 * shuffling on every render.
 */
export function entityTypeColor(entityType: string): string {
  const pinned = PALETTE[entityType];
  if (pinned !== undefined) return pinned;
  let hash = 0;
  for (let index = 0; index < entityType.length; index += 1) {
    hash = (hash * 31 + entityType.charCodeAt(index)) % 100_000_007;
  }
  return FALLBACK_COLORS[hash % FALLBACK_COLORS.length];
}

/**
 * Map a node's degree onto its drawn radius.
 *
 * Square-rooted, so *area* tracks degree: a hub with 40 edges reads as
 * bigger than one with 4 without swallowing the canvas. A slice where every
 * node has the same degree draws them all at the same middling size rather
 * than all at the minimum.
 */
export function nodeSize(degree: number, maxDegree: number): number {
  if (maxDegree <= 0) return MIN_NODE_SIZE;
  const ratio = Math.sqrt(Math.max(degree, 0) / maxDegree);
  return MIN_NODE_SIZE + (MAX_NODE_SIZE - MIN_NODE_SIZE) * ratio;
}

/**
 * Deterministic seed position for the node at `index` of `total`.
 *
 * ForceAtlas2 moves nodes by repulsion and attraction; nodes that all start
 * at the origin have no direction to be pushed in and the layout never
 * resolves. A circle is the standard seed — and a fixed one keeps the
 * finished picture reproducible.
 */
export function seedPosition(
  index: number,
  total: number,
): { x: number; y: number } {
  const angle = total <= 0 ? 0 : (2 * Math.PI * index) / total;
  return {
    x: SEED_RADIUS * Math.cos(angle),
    y: SEED_RADIUS * Math.sin(angle),
  };
}

/**
 * Turn one export into the drawable model.
 *
 * Defensive about three things graphology raises on rather than tolerates,
 * any of which would blank the whole canvas: a duplicate node id, an edge
 * pointing at a node the export did not include, and a duplicate edge key.
 * Self-loops are dropped too — they draw as nothing and say nothing.
 *
 * `degree` is the server's slice-local count (what the export drew), so the
 * size channel means "how connected inside this picture", not "in the whole
 * graph" — an export is capped, and claiming otherwise would be a lie.
 */
export function buildGraphView(graph: GraphExport): GraphViewModel {
  const seen = new Set<string>();
  const raw: GraphExportNode[] = [];
  for (const node of graph.nodes ?? []) {
    if (node.id === "" || seen.has(node.id)) continue;
    seen.add(node.id);
    raw.push(node);
  }

  const maxDegree = raw.reduce((best, node) => Math.max(best, node.degree), 0);
  const counts = new Map<string, number>();
  const nodes = raw.map((node, index) => {
    const entityType = normalizeEntityType(node.entity_type);
    counts.set(entityType, (counts.get(entityType) ?? 0) + 1);
    return {
      key: node.id,
      label: node.id,
      ...seedPosition(index, raw.length),
      size: nodeSize(node.degree, maxDegree),
      color: entityTypeColor(entityType),
      entityType,
      description: node.description ?? null,
      degree: node.degree,
    };
  });

  const edgeKeys = new Set<string>();
  const edges: GraphViewEdge[] = [];
  for (const edge of graph.edges ?? []) {
    if (edge.source === edge.target) continue;
    if (!seen.has(edge.source) || !seen.has(edge.target)) continue;
    const key = edge.id || `${edge.source}→${edge.target}`;
    if (edgeKeys.has(key)) continue;
    edgeKeys.add(key);
    edges.push({
      key,
      source: edge.source,
      target: edge.target,
      label: edge.label ?? "",
      size: 1,
      color: EDGE_COLOR,
    });
  }

  const legend = [...counts.entries()]
    .map(([entityType, count]) => ({
      entityType,
      color: entityTypeColor(entityType),
      count,
    }))
    .sort((a, b) => b.count - a.count || a.entityType.localeCompare(b.entityType));

  return { nodes, edges, legend, truncated: graph.truncated ?? false };
}

/**
 * Client-side search over the loaded nodes.
 *
 * Deliberately not a server round-trip: the whole slice is already in the
 * browser (hundreds of nodes), so filtering is instant and works while the
 * engine is busy extracting. Ranked prefix-first, the ⌘K palette's rule, so
 * typing "bob" offers Bob before "Bobbi's landlord".
 */
export function filterEntities(
  nodes: GraphViewNode[],
  query: string,
  limit = 8,
): GraphViewNode[] {
  const needle = query.trim().toLowerCase();
  if (needle === "") return [];
  const ranked: { node: GraphViewNode; rank: number; index: number }[] = [];
  nodes.forEach((node, index) => {
    const label = node.label.toLowerCase();
    if (label.startsWith(needle)) ranked.push({ node, rank: 0, index });
    else if (label.includes(needle)) ranked.push({ node, rank: 1, index });
  });
  ranked.sort((a, b) => a.rank - b.rank || a.index - b.index);
  return ranked.slice(0, limit).map((entry) => entry.node);
}

/**
 * The node keys to keep lit when one node is focused: itself plus every
 * neighbour it shares an edge with. Everything else dims — the standard
 * "what is this connected to?" affordance, computed here so the renderer
 * stays a lookup.
 */
export function neighborhood(
  edges: GraphViewEdge[],
  key: string | null,
): Set<string> {
  const lit = new Set<string>();
  if (key === null) return lit;
  lit.add(key);
  for (const edge of edges) {
    if (edge.source === key) lit.add(edge.target);
    else if (edge.target === key) lit.add(edge.source);
  }
  return lit;
}
