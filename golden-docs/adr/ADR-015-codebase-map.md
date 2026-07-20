# ADR-015: In-app codebase map (`/map` + developer mode)

**Status:** Accepted (2026-07-19) · Amended twice (2026-07-20 — see
[the canvas amendment](#amendment-2026-07-20-the-condensed-foglamp-style-canvas)
and [the layout-engine amendment](#amendment-2026-07-20-elk-layout-precomputed-beams))

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

## Amendment (2026-07-20): the condensed, foglamp-style canvas

The first shipped canvas was correct but **noisy**: 41 fully-drawn nodes and
59 edges — model fan-in arrows above all — read as a wiring diagram rather
than a map. A foglamp.dev scan of this repo rendered the same system far more
legibly, so the map was rebuilt to that standard. What changed, and why:

- **The data is now the condensed scan graph** (26 nodes / 38 edges, from
  `.foglamp/scan.json` 2026-07-20): one node per moving part instead of one
  per registry entry, with `Ingestion` / `Query path` / `Observability` as the
  three groups. Same schema discipline (`satisfies CodebaseMap`, validator,
  drift-guarded `sourceRef`s — all 15 verified against the tree); the `top*`
  callouts became standalone `{id, label, domain}` rows feeding the new side
  panel rather than node references.

- **Sink `model` nodes fold into chips.** The layout removes each model that
  has no outgoing edges and re-expresses every edge into it as a favicon chip
  on the calling card ("Retriever registry" carries `multilingual-e5-large` +
  `bge-reranker-v2-m3`). This is the single change that removes most of the
  visual noise — model usage reads as a badge, not as six arrows converging on
  three boxes. A model with outgoing edges would stay a card, so nothing can
  silently disappear.

- **HTML cards over one SVG edge underlay** replace the all-SVG canvas: nodes
  are real absolutely-positioned `<button>` cards inside a single transformed
  "world" div (native focus/activation — the SVG `role="button"` workarounds
  are gone), while edges stay SVG: rounded-orthogonal polylines with chevron
  arrowheads, routed through corridor channels and free horizontal bands so
  long edges travel open air instead of slicing through groups. Groups render
  as containers with an internal top-down mini-layout (no longer bands behind
  columns — the global pass treats each group as one super-node).

- **Favicons are vendored, not fetched.** The original "favicons omitted"
  stance protected the no-phone-home rule at the cost of the look. The
  amendment keeps the rule and the look: the eleven integration/model icons
  were fetched once at authoring time into `web/public/map-icons/` and served
  same-origin, with the kind glyph as the on-error fallback (the
  privacy-plugin-self-hosts-fonts precedent from the docs site).

- **Kind identity follows the reference design**: colored icon-badge squircles
  (bolt / ghost / hexagon / database / world — Tabler's MIT filled glyphs,
  inlined) plus a floating Models/Integrations panel, a bottom legend pill bar
  whose entries spotlight their kind on hover, an anchored detail popover, and
  a slow "beam" border on agent cards (static under reduced motion).

Unchanged: the route + developer-mode gating, the curated-data stance and
update rule, the Vitest/Playwright testing split, the drift guard, and the
DAG invariant. The counts in the Consequences above describe the original
graph; the shipped graph is 26/38 with the same 60/120 caps.

## Amendment (2026-07-20): ELK layout, precomputed; beams

The first amendment's hand-rolled router got the vocabulary right but not
the composition: the reference (foglamp.dev's scan viewer) reads better
because of *placement* decisions no reasonable hand-rolled pass reproduces.
Its actual renderer (foglamp-labs/foglamp, Apache-2.0) was reviewed and
ported:

- **ELK's layered algorithm replaces the hand-rolled layout** — reversing
  this ADR's "no graph library" decision, with its premise changed: the
  requirement moved from "a deterministic layered layout" to "*this exact*
  layout". Two passes (each group top-to-bottom in isolation, then the root
  graph left-to-right over group boxes with `BRANDES_KOEPF`/`BALANCED`
  placement and orthogonal routing), ELK-reserved inline edge labels (they
  can never collide), cross-group edges attached to the group container and
  deduped per endpoint pair (`orig` indices keep the originals for
  trace/dim), then a row snap and an empty-band squeeze
  (`web/lib/map-layout.ts`, ported with attribution).

- **elkjs never ships to the browser.** The layout of a static input is
  itself static, so the ELK pass runs at authoring time into a checked-in
  snapshot (`web/lib/codebase-map.layout.json`) that the page imports
  synchronously — the `openapi.json` pattern again: a Vitest guard re-runs
  ELK and fails CI on drift; `bun run test -u` regenerates. This also fixed
  a real defect of the first attempt at runtime ELK (`use()` + Suspense):
  the server-rendered map was *visible* before the client had fetched the
  elk chunk and hydrated, so a fast first click landed on a dead card.
  Instant hydration closed that window (and Playwright's activation tests
  are what caught it).

- **The traveling beams** are the reference's signature liveliness and are
  now ported: per edge, a comet (gradient pill on a CSS `offset-path`,
  WAAPI-animated at constant speed) occasionally runs the route and flashes
  the target card's ring in the source kind's color — sparse by design (one
  run, ~30–55 s rest, staggered starts), compositor-friendly, skipped
  entirely under reduced motion. Entrance choreography (cards spring in
  along the flow direction, edges draw themselves) rides plain CSS
  keyframes with `animation-fill-mode: backwards`, so the stagger can never
  fight the spotlight's dim classes. Wheel now pans (pinch/ctrl zooms),
  matching the reference's map-app gesture model; keyboard zoom stays.

Consequences: elkjs is a dependency but a build/test-time one in practice
(its dynamic-import chunk is unreachable from page code); the layout module
is a port, so upstream improvements are easy to re-take; and the map gains a
second generated artifact whose regeneration command sits next to its guard.
