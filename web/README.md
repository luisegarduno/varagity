# Varagity web GUI

The browser front-end of Varagity — a Next.js app, and the only
browser-facing surface of the stack. It holds no pipeline logic: every
request goes to the FastAPI backend on `:8000` (JSON + SSE chat streaming),
which runs the same Prefect flows as the CLI.

## Toolchain

- **bun** — the package manager (`bun install`, `bun run <script>`, `bunx`);
  the lockfile is `bun.lock`.
- **Node** — the runtime: Next.js, Vitest, and Playwright all execute under
  Node via their shebang scripts (no `--bun`, no `bun test`).
- **Vitest** — unit tests, coverage floors enforced in CI.
- **Playwright** — opt-in e2e against the live stack.

## Commands

```bash
bun install          # install dependencies (bun.lock)
bun run dev          # dev server on :3000 — set NEXT_PUBLIC_API_URL first
bun run test         # Vitest unit tests (coverage floors)
bun run lint         # eslint
bun run build        # production build
bun run e2e          # opt-in Playwright — needs the live stack on :3000/:8000
bun run gen:types    # regenerate lib/types.ts from the API's OpenAPI schema
```

- `NEXT_PUBLIC_API_URL` is inlined by Next.js at build/dev time; in Docker it
  is a **build-time** constant (a compose build arg) — changing it requires
  `docker compose build web`.
- Before the first `bun run e2e`, install the browsers explicitly:
  `bunx playwright install --with-deps` (no lifecycle script does it for you).
- `lib/types.ts` is generated — never hand-edit it.

## Read first

- [`AGENTS.md`](AGENTS.md) — this Next.js version post-dates model training
  data; read `node_modules/next/dist/docs/` before writing any code.
- [`../golden-docs/`](../golden-docs/index.md) — as-built architecture, the
  HTTP API contract, and the runbook (`uv run mkdocs serve` at the repo root).
