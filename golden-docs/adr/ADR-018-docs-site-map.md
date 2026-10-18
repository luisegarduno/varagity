# ADR-018: The codebase map on the docs site

**Status:** Accepted (2026-10-18) · Builds on
[ADR-015](ADR-015-codebase-map.md) (the map) and respects
[ADR-014](ADR-014-bun-package-manager.md) (bun is the package manager only)

## Context

[ADR-015](ADR-015-codebase-map.md) put the one picture of the whole system
at `/map` in the web GUI — which means it exists only while the stack is up.
The place that *explains* the system, this site (`golden-docs/`, published
to GitHub Pages), had no picture at all. The ask was the map as a docs page
**with its animations intact**: the entrance choreography, the agent-card
beam, and the traveling comets are the point of the foglamp treatment, not
decoration, and a screenshot or a Mermaid redraw would have lost the trace,
the popovers, and the motion together.

Four constraints shaped the answer. The published site must keep working
offline and never phone home (the privacy-plugin precedent in `mkdocs.yml`).
The Python docs job in CI must not grow a JavaScript toolchain. ADR-014's
line — bun installs, Node executes — stands. And there must remain exactly
one implementation of the map: a port that could drift from `/map` would
be a second thing to keep true.

## Decision

- **Same component, second build.** `web/docs-map/` holds a page shell
  (`index.html`, `main.tsx`, `map.css`, `fonts.css`) that mounts the app's
  own `CodebaseMapView` outside the Next shell, plus its Vite config and the
  manifest helper. `vite build` (Vite with
  `@tailwindcss/vite`, run under Node by `bun run build:docs-map`) emits a
  self-contained bundle into `golden-docs/assets/map/`: one page, one
  script, one stylesheet, the vendored favicons, the fonts stylesheet, and
  a manifest. `map.css` imports the app's `globals.css` whole, so the docs
  page and `/map` cannot drift apart on colour or motion; Tailwind's source
  detection runs from the Vite root and the map's component files are named
  explicitly, so only the map's utilities are emitted.

- **A committed, drift-guarded artifact** — the `openapi.json` pattern
  (`golden-docs/openapi.json`, `web/lib/codebase-map.layout.json`). The
  bundle is checked in; `manifest.json` records a sha256 over its **inputs**
  (the map components, data, layout snapshot, `globals.css`, the icons, and
  `web/docs-map/` itself); `web/lib/__tests__/docs-map.test.ts` recomputes
  that hash and fails CI when the committed bundle was built from anything
  else. Inputs rather than output bytes, because minifier output is not
  byte-stable across tool versions; dependency versions deliberately
  excluded, because a React patch changes the bytes but not the picture.
  The same test pins the properties the site relies on: relative URLs, the
  icons beside the page, one script and no ELK, and nothing in the bundle
  for the privacy plugin to fetch except the fonts it is meant to localize.

- **Framed, not inlined.** `golden-docs/map.md` embeds the page in a
  same-origin `<iframe>` (plus a full-screen link). Theme follows the site:
  an inline script in `index.html` reads Material's
  `<body data-md-color-scheme>` through the parent, mirrors it onto the
  frame's `.dark` class, and watches it with a `MutationObserver`; opened
  standalone (or framed by a page it cannot read) it follows the OS. It runs
  ahead of the stylesheet, so the first paint is already the right scheme.

- **Relative everything.** GitHub Pages serves the site under `/varagity/`,
  so the build uses `base: "./"` and the favicons gained an indirection:
  `MapIconBaseContext` (`web/components/map/icon-base.ts`) defaults to the
  app's absolute `/map-icons`, and the docs entry provides `map-icons`,
  resolved beside the page. Nothing else in the map knows the path.

- **Fonts through the privacy plugin.** `next/font` is not available outside
  Next, so `fonts.css` carries one `@import url(https://fonts.googleapis.com/…)`
  for Geist, Geist Mono, and Instrument Serif. Material's privacy plugin
  rewrites external `url()`s it finds in static stylesheets and downloads
  the files at build time, exactly as it does for the site's own
  `extra_css` entry — so the published page never contacts Google Fonts. A
  `<link>` in the static HTML would **not** be rewritten (the plugin
  processes rendered pages and static `.css`/`.js`, not static `.html`),
  which is why the import lives in a stylesheet.

## Rationale

The alternatives, each rejected for a reason worth keeping:

- **Frame the live app** (`<iframe src="http://localhost:3000/map">`). Needs
  the stack; every Pages reader gets a dead frame.
- **Next static export.** `output` is config-global (the Docker image needs
  `standalone`), `/c/[id]` cannot be enumerated, and the export would drag
  the sidebar shell along.
- **Mount the bundle inline in the Material page.** Tailwind's preflight and
  Material's cascade would fight across the whole page — `varagity.css`'s
  `.md-typeset h1/h2` rules alone would restyle the side panel — and
  `navigation.instant`'s XHR page swaps would need remount choreography.
  The iframe is an isolation boundary that costs nothing here.
- **A static SVG, a Mermaid redraw, or a screenshot.** Loses the trace, the
  detail popovers, and the motion — the explicit ask.
- **A vanilla-JS port.** A second implementation of the map to keep true.
- **Build at docs time** (a MkDocs hook running the bundler). Puts bun and
  Node in the Python CI job, and misbehaves whenever they are absent.
- **`bun build`.** Already in the binary and fast, but it is the bun
  *runtime* bundling — the line ADR-014 drew. Vite was already in the tree
  as Vitest's peer; promoting it to a declared devDependency cost nothing.

## Consequences

- **A third generated artifact and its command.** `bun run build:docs-map`
  joins `scripts/export_openapi.py` and `bun run test -u`; the update rule
  in [Architecture → The codebase map](../architecture.md#the-codebase-map)
  has a second step. Sizes at adoption: ≈290 KB of script (88 KB gzipped —
  React is in it), ≈49 KB of stylesheet (9 KB gzipped), 46 KB of icons; a
  regeneration adds on the order of 100 KB compressed to history, and
  regenerations track map changes (three in the map's first three months).
- **`vite` and `@tailwindcss/vite` are devDependencies.** The Python docs
  job is unchanged: `mkdocs build --strict` copies the committed bundle
  verbatim, and the Vitest guard runs in the web job.
- **The in-app map is functionally untouched** — the icon path moved behind
  a context with the old default. Playwright's `/map` specs still apply.
- **Known behaviours, by design.** The wheel pans inside the frame rather
  than scrolling the page (standard embedded-map behaviour, said on the
  page; the frame is bounded and the full-screen link exists). And the guard
  shares ADR-015's residual risk: it cannot tell whether an edge still tells
  the truth — edge semantics stay a human-maintained property.
