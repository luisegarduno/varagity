# ADR-014: bun as the package manager, Node as the runtime

**Status:** Accepted (2026-07-19)

## Context

`web/` ran pnpm 10 for dependency management with Node 22 as the runtime.
v3 (spec_v3 §7) consolidates on **bun 1.3.14** — but bun is two products
in one binary: a package manager and a JavaScript runtime/test runner.
The decision that needed recording is where the line sits, because "use
bun" without a scope line invites `bun --bun next build` and `bun test`
by drift, and those change what executes in production and what gates
coverage. [ADR-005's amendment](ADR-005-web-stack-and-api.md#amendment-2026-07-15-pnpm-bun)
records *that* the migration happened; this ADR records the
scope line and why it sits where it does.

## Decision

- **bun is the package manager only.** `bun install` owns dependency
  resolution (`web/bun.lock`); every script still executes **under Node**
  via shebang delegation — `bun run build` invokes the same `next`
  binary whose `#!/usr/bin/env node` line picks Node 22. No `--bun`, no
  `bun test`, no runtime swap anywhere: `web/Dockerfile`'s
  `base`/`deps`/`build` stages moved to `oven/bun:1.3.14-alpine`, and the
  `run` stage **stays `node:22-alpine`** executing `node server.js`.
- **Vitest stays the test runner**, with its four coverage floors
  unchanged. `bun test` was rejected on three verified grounds: it
  ignores `vitest.config.ts`, its mocking API differs, and — decisive —
  **its coverage reporting has no threshold enforcement**, which would
  silently drop the floors CI gates on.
- **The runtime swap is deferred behind two unresolved upstream risks**
  that land on this exact stack: `sharp` under bun on Alpine in CI
  (lovell/sharp#4215, unresolved) and Next's `output: "standalone"`
  under the bun runtime (no official word). Recorded as a post-v3 spike,
  not a v3 stretch goal.
- **`trustedDependencies` was translated by polarity, not mechanically.**
  pnpm's `ignoredBuiltDependencies` is a default-allow *blocklist*; bun's
  `trustedDependencies` is a default-deny *allowlist* over a curated
  built-in list. `sharp` sits on bun's default-trusted list — its
  postinstall now runs where pnpm blocked it (the vendor default for
  Next's own image library, and strictly required for `next/image`).
  `unrs-resolver` is not default-trusted; it is allowlisted explicitly in
  `web/package.json` as cheap insurance against the known
  "failed to load native binding" lint failure. Both are transitive
  dependencies — neither appears in `dependencies` at all.
- **CI pins both coordinates the spec conflated**:
  `oven-sh/setup-bun@v2.2.0` is the *action* tag; `bun-version: 1.3.14`
  is the *bun* release — there is no bun 2.x. `actions/setup-node@v4`
  is **kept** at `node-version: 22` (matching the Dockerfile `run`
  stage) because Node is still what executes Vitest and Next; its
  pnpm-specific cache inputs are gone, replaced by an explicit
  `actions/cache` on `~/.bun/install/cache` (`setup-bun` caches only the
  bun binary, and `setup-node` has no bun cache type).
- **The lockfile conversion is one-way and must stay one-way**:
  `bun install` auto-converts a `pnpm-lock.yaml` **once, on `bun.lock`
  absence** — so `pnpm-lock.yaml` and `pnpm-workspace.yaml` were deleted
  in the same commit that added `bun.lock`. Two lockfiles in the repo
  means the next contributor's install silently re-converts a stale
  file. The `"packageManager"` field was deleted too — corepack has no
  bun support, and bun neither reads nor writes it.

## Rationale

- **The migration's value is install speed and one fewer tool**, not
  runtime performance: a cold-cache `bun install --frozen-lockfile`
  measured **~3.2 s for 779 packages**. This app's latency is dominated
  by two local GPUs; swapping the JS runtime buys nothing measurable and
  carries the two open risks above — which is why the aggressive shape
  doesn't clear the bar.
- **Proven equivalence before the cutover**: the bun-resolved dependency
  tree was diffed against pnpm's — 868/868 packages identical — so the
  swap changed the installer, not the installed.
- **Playwright's browser install became a documented step**
  (`bunx playwright install --with-deps` in the runbook): no lifecycle
  script ever installed browsers under pnpm either; the migration just
  made the implicit explicit.

## Consequences

- One JS toolchain command surface: `bun install` / `bun run <script>` /
  `bunx …` — never `npm`/`yarn`/`pnpm`. Scripts stayed byte-identical,
  including the POSIX `${VAR:-default}` expansion in `gen:types` (on
  Linux `bun run` delegates to a real shell).
- `sharp`'s postinstall now runs. If it ever misbehaves, bun has no
  blocklist to reach for — the recorded fallback is pinning the
  `@img/sharp-*` platform packages explicitly.
- The web CI job caches by `bun.lock` hash; a lockfile change is the
  cache key change.
- Node remains a hard prerequisite alongside bun (contributors need
  both; the Dockerfile stages encode the same split).
- Revisit trigger, recorded: the runtime spike (`bun --bun`, `bun test`)
  becomes worth re-opening if sharp#4215 resolves *and* Vitest coverage
  thresholds land in `bun test` — both, not either.
