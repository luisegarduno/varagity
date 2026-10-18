---
hide:
  - navigation
  - toc
---

# Codebase map

The whole system as one interactive graph — the same map the web GUI serves
at `/map` ([ADR-015](adr/ADR-015-codebase-map.md)), built into this site so
it reads without the stack running ([ADR-018](adr/ADR-018-docs-site-map.md)).
Entry points, the Prefect flows, the chat and graph engines, the pluggable
registries, and the datastores are cards; the three models fold into chips on
the cards that call them; the Ingestion / Query path / Graph corpus /
Observability groups draw as containers. It puts the *relational* facts on
the edges where they belong — "`reranked` composes a base retriever",
"condense calls the same llama.cpp server the answer generator does".

**Click a card** to spotlight its downstream flow and open its details;
<kbd>Esc</kbd> or a click on the background clears. **Drag** to pan (the
scroll wheel pans too); pinch or <kbd>Ctrl</kbd> + wheel zooms;
<kbd>+</kbd> / <kbd>−</kbd> / <kbd>0</kbd> zoom and reset from the keyboard,
and <kbd>Tab</kbd> walks the cards in reading order. Hovering a legend entry
highlights that kind. Cards and edges enter along the flow direction, and
every so often a comet runs an edge and flashes the card it lands on — the
same motion as the app, and still under `prefers-reduced-motion`. The map
follows this site's light/dark toggle.

<!-- Raw HTML is not path-rewritten by MkDocs: this page renders at map/
     (directory URLs), so the frame's path climbs one level. The Markdown
     link below is rewritten, so it stays docs-relative. -->
<div class="codebase-map">
<iframe src="../assets/map/index.html" title="Varagity codebase map — interactive"></iframe>
</div>

[:material-arrow-expand-all: Open the map full screen](assets/map/index.html){ .md-button }

## How this page is built

The frame is not a screenshot and not a second implementation: it is the
app's own `CodebaseMapView` — the curated graph in `web/lib/codebase-map.data.ts`,
the checked-in ELK layout snapshot, the entrance choreography, the agent-card
beams and the traveling comets — bundled by Vite from `web/docs-map/` into
`golden-docs/assets/map/` and committed, the way `openapi.json` is. Only two
things differ from `/map`: the favicons are served beside the page (GitHub
Pages serves this site under a repository subpath), and the fonts arrive
through a stylesheet the privacy plugin localizes at build time, so the
published page never contacts Google Fonts.

**Update rule.** The map stays true by hand, in two steps. Edit
`web/lib/codebase-map.data.ts` when the architecture changes (the rule in
[Architecture → The codebase map](architecture.md#the-codebase-map)), then run

```bash
cd web && bun run build:docs-map
```

and commit `golden-docs/assets/map/` with it. A Vitest guard
(`web/lib/__tests__/docs-map.test.ts`) hashes the bundle's inputs — the map
components, data, layout, `globals.css`, the icons, and `web/docs-map/` —
and fails CI when the committed bundle was built from anything else. What
it cannot catch is the same thing the in-app guard cannot: whether an edge
still tells the truth.
