import {
  expect,
  type APIRequestContext,
  type Page,
  type TestInfo,
} from "@playwright/test";

/** The two themes the a11y criterion covers. */
export type ThemeName = "light" | "dark";

/** The two layout densities (mirrors `lib/display-prefs.ts` DENSITIES). */
export type DensityName = "comfortable" | "compact";

/** Matches the conversation route the app redirects to (unhyphenated UUID hex). */
export const CONVERSATION_URL = /\/c\/[0-9a-f-]{32,36}/;

/**
 * Prime localStorage before any app script runs. Must be called before the
 * first `page.goto`.
 *
 * - `theme`: next-themes' key — forces light/dark deterministically
 *   (unset ⇒ "system", which follows the browser's emulated color scheme).
 * - `density`: the display pref (`varagity:density`) — set to exercise the
 *   map's density-sensitive layout; unset leaves the "comfortable" default.
 * - `developerMode`: the cosmetic gate (`varagity:developer-mode`) — set
 *   `false` to hide the Map sidebar button and the ⌘K command; unset leaves
 *   the default-on behavior (`getItem() !== "false"`).
 * - The evidence rail pref is pinned open so desktop runs are deterministic
 *   regardless of what a previous session collapsed.
 */
export async function primeAppState(
  page: Page,
  opts: {
    theme?: ThemeName;
    density?: DensityName;
    developerMode?: boolean;
  } = {},
): Promise<void> {
  await page.addInitScript(
    (state: {
      theme: string | null;
      density: string | null;
      developerMode: boolean | null;
    }) => {
      if (state.theme) window.localStorage.setItem("theme", state.theme);
      if (state.density)
        window.localStorage.setItem("varagity:density", state.density);
      if (state.developerMode !== null)
        window.localStorage.setItem(
          "varagity:developer-mode",
          String(state.developerMode),
        );
      window.localStorage.setItem("varagity:evidence-rail-open", "true");
    },
    {
      theme: opts.theme ?? null,
      density: opts.density ?? null,
      developerMode: opts.developerMode ?? null,
    },
  );
}

/** True when running under the mobile project (390×844 + touch). */
export function isMobileProject(testInfo: TestInfo): boolean {
  return testInfo.project.name === "mobile-chromium";
}

/**
 * Land on the app root and wait out the client-side bootstrap redirect to
 * the newest conversation (`/c/<uuid>`).
 */
export async function gotoApp(page: Page): Promise<void> {
  await page.goto("/");
  await page.waitForURL(CONVERSATION_URL, { timeout: 30_000 });
}

/**
 * Create a brand-new (empty) conversation via the "New chat" affordance —
 * the sidebar button on desktop, the top-bar icon button below `md` (both
 * carry the accessible name "New chat"; only the visible one matches).
 * Read-only toward the corpus: conversations are fair game, documents are
 * not.
 */
export async function gotoFreshConversation(page: Page): Promise<void> {
  await gotoApp(page);
  const before = page.url();
  await page.getByRole("button", { name: "New chat" }).click();
  await page.waitForURL(
    (url) => url.href !== before && CONVERSATION_URL.test(url.pathname),
    { timeout: 15_000 },
  );
  await expect(page.getByLabel("Question")).toBeVisible();
}

/** The live API the web app talks to (override for a non-default stack). */
export const API_URL = process.env.PLAYWRIGHT_API_URL ?? "http://localhost:8000";

/**
 * Why the graph specs cannot run here, or `null` when they can.
 *
 * The graph corpus is the one thing in this repo a test may not create: a
 * real backfill runs for hours against the owner's archive. So the graph
 * specs read the live stack's state and skip cleanly when there is nothing
 * extracted — a fresh clone, a `docker volume rm varagity_graphdata`, or
 * `GRAPH_ENABLED=false` are all legitimate, and none of them is a failure
 * of the code under test.
 *
 * Specs that spend LLM time pass `requireIdle` — a running build owns the
 * single llama.cpp slot, so a graph turn queues behind extraction calls
 * with no latency bound (the runbook's query-during-build contention).
 * Read-only specs (tabs, view/export) stay runnable mid-build.
 *
 * @returns The skip reason, or `null` when a built graph is available.
 */
export async function graphSkipReason(
  request: APIRequestContext,
  opts: { requireIdle?: boolean } = {},
): Promise<string | null> {
  let status: {
    enabled?: boolean;
    building?: boolean;
    entities?: number | null;
    documents?: Record<string, number>;
  };
  try {
    const response = await request.get(`${API_URL}/api/graph/status`);
    if (!response.ok()) return `GET /api/graph/status → ${response.status()}`;
    status = await response.json();
  } catch {
    return `the API at ${API_URL} is unreachable — is the stack up?`;
  }
  if (status.enabled === false) {
    return "GRAPH_ENABLED is false on this stack";
  }
  if (opts.requireIdle && status.building === true) {
    return "a graph build is running — turn latency is unbounded behind the single llama.cpp slot; re-run when it finishes";
  }
  const built =
    (status.entities ?? 0) > 0 ||
    Object.values(status.documents ?? {}).some((count) => count > 0);
  return built
    ? null
    : "the live graph is empty — upload a message archive and run a build first";
}

/** Assert the html element reflects the forced theme (next-themes class strategy). */
export async function expectTheme(page: Page, theme: ThemeName): Promise<void> {
  const html = page.locator("html");
  if (theme === "dark") {
    await expect(html).toHaveClass(/(^|\s)dark(\s|$)/);
  } else {
    await expect(html).not.toHaveClass(/(^|\s)dark(\s|$)/);
  }
}
