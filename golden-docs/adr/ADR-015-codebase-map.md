# ADR-015: In-app codebase map (`/map` + developer mode)

**Status:** Accepted (2026-07-19)

## Context

`golden-docs/` explains Varagity in prose and Mermaid fragments, but nothing
shows the whole system at once, and the interesting facts about this codebase
are *relational*: "the `reranked` retriever composes a base retriever rather
than forking fusion", "the condense engine calls the same llama.cpp server the
answer generator does", "chunks live in both stores joined by
`(doc_id, original_index)`". Those are edges, not paragraphs.

The ask was an in-app **map of how Varagity works** — entries, the Prefect
flows, the chat engines, the three models, the pluggable registries, and the
datastores — reachable from a new **developer mode** toggle. A cluster of
design questions had to be settled before writing it, and they are worth
recording because several are counter-intuitive (JSON *loses* the type safety
it looks like it should give; a graph library *costs* more than it saves at
this size). This is the first ADR scoped to the frontend (`web/`).

## Decision

- **A route (`/map`), not an overlay.** Corpus is already a route; a map wants
  a full-viewport canvas (a dialog's scroll lock fights pan/zoom), and a route
  is deep-linkable and ⌘K-navigable. It renders inside the existing shell as a
  thin server `page.tsx` (the house route shape) mounting one client view.

- **Curated static data, never live introspection.** The graph is a
  hand-maintained artifact (`web/lib/codebase-map.data.ts`), not the output of
  an AST scan or a build step. Curation is the value here — an auto-generated
  import graph would be strictly worse, and it would drag runtime/build
  machinery into a documentation feature. The map is updated by a human when
  the architecture changes, like `golden-docs/` itself.

- **TypeScript `… satisfies CodebaseMap`, not JSON.** The obvious choice —
  `codebase-map.json`, imported for build-time type-checking — does **not**
  actually type-check the `kind`/`EdgeKind` string unions: a JSON module's
  string literals widen to `string` before the checker ever compares them to
  the union (microsoft/TypeScript#26552). Authoring the data as a `.ts` module
  ending in `satisfies CodebaseMap` makes the union check real (an invalid
  `kind` fails `bun run typecheck`) while keeping the file hand-editable, with
  no runtime request and no `openapi.json`-style wire surface.

- **A hand-rolled SVG canvas, no graph library.** Nothing in `web/`'s
  dependencies fits, and the layout+interaction model (deterministic layered
  layout, pan, zoom-to-cursor, click-to-trace) is a few hundred lines of pure
  functions and event handlers. The candidates cost more than they save at
  ~41 nodes: dagre ≈13 KB gzipped, d3-dag ≈42 KB, elkjs ≈433 KB — all for a
  layout we can compute deterministically ourselves and unit-test without a
  DOM. A force simulation was rejected outright: a wobbling map is a bad map.

- **Brackets, not columns, for groups.** Each node keeps its own longest-path
  rank; a group renders as a labeled band drawn *behind* its members wherever
  they land, so a group may span adjacent columns. The spec's original rule
  ("pull grouped nodes into one column") collapsed a dozen edges, because no
  group in the corrected graph is an antichain. Bands-behind-members keeps
  every edge renderable and keeps the trace feature honest
  (`flow-ingest → parsers`, etc.). The corrected graph is a DAG, enforced as a
  map invariant so the rank step can rely on it.

- **localStorage cosmetic gating, not a server pref or a route guard.**
  Developer mode is a client-only display pref (`varagity:developer-mode`),
  stored the same way as accent/density/evidence-rail and defaulting **on**
  (absence reads as enabled, `getItem() !== "false"`). It hides the sidebar
  Map button and the ⌘K command; it does **not** guard the route — `/map`
  stays reachable by URL. A hard guard needs a redirect and an SSR-safe read
  for no security benefit in a single-user local app. Said so in the docstring.

- **Favicons omitted.** A favicon fetch is a third-party request, and this app
  never phones home; `domain` renders as a small monospace tag instead.

## Rationale

- **The map's whole value is that it stays true.** The `sourceRef` drift guard
  (invariant 7) resolves every referenced path against the repo root in Vitest
  and fails CI on a rename or a file that shrinks past a pinned line — the same
  posture as the drift-guarded `openapi.json` snapshot. Min-count assertions
  ("guard-of-the-guards", the `test_dashboards.py` idiom) keep that guard from
  passing vacuously if the data ever fails to load. Two paths that can never
  resolve on a fresh CI checkout carry **no** `sourceRef`: the gitignored
  `DOCS_PATH` corpus, and — until its own page existed — the map route itself
  (added in the same commit that created `web/app/map/page.tsx`, guard and
  artifact together, per the `openapi.json` precedent).

- **The house testing split is deliberate.** Vitest stays lib-only pure logic
  (per its config policy): the map invariants, the drift guard, and the layout
  properties are all DOM-free and coverage-counted. Everything that renders —
  node presence, trace/detail interaction, keyboard operation, the gating, and
  axe (critical) across themes × densities — is Playwright, opt-in against the
  live stack. No jsdom, no testing-library, no component-level Vitest.

- **The data ships corrected, not transcribed.** An as-built verification of
  the drafted graph found seven material divergences (four reversed edges, a
  node miscount, a keep-10-vs-5 label error, and the group-column collapse
  above) plus smaller edge fixes; the checked-in data resolves all of them, so
  the map matches the code as built rather than the spec's first draft.

## Consequences

- **The update rule is now documented** next to the OpenAPI-snapshot rule it
  mirrors (`architecture.md` → "The codebase map"): edit
  `web/lib/codebase-map.data.ts` when the architecture changes. The drift guard
  catches renames and truncation; it **cannot** catch *semantic* drift — an
  edge that quietly stops being true keeps passing CI. Edge truth is a
  human-maintained property, called out as accepted residual risk.

- **41 nodes / 59 edges, well under the caps** (60 nodes / 120 edges). The
  layout is O(V+E) and runs once per mount over a static import; the data file
  ships in the `/map` client bundle only. The caps exist so no performance work
  is ever warranted.

- **First frontend-scoped ADR.** Prior ADRs record backend and ops decisions;
  this one establishes that `web/`-only decisions with lasting consequences get
  recorded here too.

- **Developer mode is a seam for future developer surfaces**, not just the map
  — the generic `developer mode` toggle and its default-on stance leave room to
  gate more behind it without a new preference.
