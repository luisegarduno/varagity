import { defineConfig } from "@playwright/test";

/**
 * Opt-in e2e harness. Deliberately NO `webServer` block: these
 * specs exercise the real stack — the web container on :3000 talking to the
 * live API on :8000 with GPU services behind it — so bring it up first
 * (`docker compose up -d --wait`) and run `bun run e2e` when you want the
 * full-flow + axe gates. Vitest (`bun run test`) never picks these up: its
 * include is scoped to lib/ + components/, and this config owns ./e2e.
 */
export default defineConfig({
  testDir: "./e2e",
  // Single-user backend (one conversation store, one GPU): keep runs serial.
  fullyParallel: false,
  workers: 1,
  // LLM generation on the local GPUs runs 30–90 s per answer.
  timeout: 120_000,
  expect: { timeout: 15_000 },
  reporter: [["list"]],
  use: {
    baseURL: process.env.PLAYWRIGHT_BASE_URL ?? "http://localhost:3000",
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "desktop-chromium",
      use: { viewport: { width: 1280, height: 800 } },
    },
    {
      name: "mobile-chromium",
      use: {
        viewport: { width: 390, height: 844 },
        isMobile: true,
        hasTouch: true,
      },
    },
  ],
});
