import path from "node:path";

import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    alias: { "@": path.resolve(import.meta.dirname) },
  },
  test: {
    environment: "node",
    include: ["lib/**/*.test.ts", "components/**/*.test.ts"],
    // Coverage floor for the unit-testable pure-logic layer (plan decision
    // #15): lib/ only — components are exercised by the opt-in Playwright
    // harness against the live stack, not by Vitest. lib/types.ts is
    // generated (pnpm gen:types) and lib/__tests__/ is the suite itself.
    // Thresholds sit ~5–10 points below measured totals (≈68% stmts/lines,
    // ≈76% branches, ≈58% funcs — untested event buses & display-prefs drag
    // the averages), mirroring the Python 80%-vs-~87% posture.
    coverage: {
      provider: "v8",
      include: ["lib/**/*.ts"],
      exclude: ["lib/types.ts", "lib/__tests__/**"],
      thresholds: {
        statements: 60,
        branches: 70,
        functions: 50,
        lines: 60,
      },
    },
  },
});
