import { expect, type Page, type TestInfo } from "@playwright/test";

/** The two themes the a11y criterion covers. */
export type ThemeName = "light" | "dark";

/** Matches the conversation route the app redirects to (unhyphenated UUID hex). */
export const CONVERSATION_URL = /\/c\/[0-9a-f-]{32,36}/;

/**
 * Prime localStorage before any app script runs. Must be called before the
 * first `page.goto`.
 *
 * - `theme`: next-themes' key — forces light/dark deterministically
 *   (unset ⇒ "system", which follows the browser's emulated color scheme).
 * - The evidence rail pref is pinned open so desktop runs are deterministic
 *   regardless of what a previous session collapsed.
 */
export async function primeAppState(
  page: Page,
  opts: { theme?: ThemeName } = {},
): Promise<void> {
  await page.addInitScript((theme) => {
    if (theme) window.localStorage.setItem("theme", theme);
    window.localStorage.setItem("varagity:evidence-rail-open", "true");
  }, opts.theme ?? null);
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

/** Assert the html element reflects the forced theme (next-themes class strategy). */
export async function expectTheme(page: Page, theme: ThemeName): Promise<void> {
  const html = page.locator("html");
  if (theme === "dark") {
    await expect(html).toHaveClass(/(^|\s)dark(\s|$)/);
  } else {
    await expect(html).not.toHaveClass(/(^|\s)dark(\s|$)/);
  }
}
