import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page, type TestInfo } from "@playwright/test";

import { CODEBASE_MAP } from "../lib/codebase-map.data";
import { condense } from "../lib/map-layout";
import {
  gotoApp,
  isMobileProject,
  primeAppState,
  type DensityName,
  type ThemeName,
} from "./helpers";

/**
 * The codebase map's e2e coverage (map Phase 5, updated for the
 * foglamp-style canvas). It absorbs the two component tests the spec
 * sketched per owner decision #2 — Vitest stays lib-only pure logic, so
 * everything that actually renders is asserted here against the live stack.
 *
 * The map itself needs no API (it renders a static import), but the discovery
 * paths — sidebar button and ⌘K command — bootstrap through the app shell, so
 * this spec runs under the same opt-in harness as the rest of `e2e/`.
 */

// Scan the settled design, not a transition frame (the a11y.spec rationale):
// reduced motion also makes the spotlight opacity flip instant, so the CSS
// assertions below never race a 300ms fade.
test.use({ contextOptions: { reducedMotion: "reduce" } });

const THEMES: readonly ThemeName[] = ["light", "dark"];
const DENSITIES: readonly DensityName[] = ["comfortable", "compact"];

/** The canvas region — `aria-label="Codebase map"` on the map `<section>`. */
const CANVAS = 'section[aria-label="Codebase map"]';

/** The nodes that render as cards (sink models fold into chips). */
const RENDERED_NODES = condense(CODEBASE_MAP).nodes;

/** Escape a node label for use inside a `^label,` accessible-name regex. */
function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Scan the current page with axe and gate on CRITICAL violations only (the
 * house criterion). Serious/moderate are left to the global a11y sweep; this
 * spec's job is to add the density axis over `/map`, not to re-police contrast.
 */
async function expectNoCriticalA11y(
  page: Page,
  testInfo: TestInfo,
  label: string,
): Promise<void> {
  const results = await new AxeBuilder({ page }).analyze();
  const critical = results.violations.filter(
    (violation) => violation.impact === "critical",
  );
  if (results.violations.length > 0) {
    await testInfo.attach(`axe-map-${label.replaceAll("/", "-")}`, {
      body: JSON.stringify(results.violations, null, 2),
      contentType: "application/json",
    });
  }
  expect(
    critical,
    `critical axe violations on ${label}:\n${JSON.stringify(critical, null, 2)}`,
  ).toEqual([]);
}

test("URL: /map renders every card in the condensed graph", async ({ page }) => {
  await primeAppState(page);
  await page.goto("/map");

  const canvas = page.locator(CANVAS);
  await expect(canvas).toBeVisible();
  await expect(page.getByRole("heading", { name: /codebase map/i })).toBeVisible();

  // One card per rendered node — the count stays in lock-step with the data
  // because it is derived from the same import (models fold into chips).
  const nodes = canvas.locator("[data-map-node]");
  await expect(nodes).toHaveCount(RENDERED_NODES.length);

  // …and each carries its own label ("{label}, {kind}").
  for (const node of RENDERED_NODES) {
    await expect(
      canvas.getByRole("button", {
        name: new RegExp(`^${escapeRegExp(node.label)},`),
      }),
    ).toBeAttached();
  }

  // The folded models surface as chips on their calling cards instead.
  await expect(
    canvas.getByRole("button", { name: /^Retriever registry,/ }),
  ).toContainText("bge-reranker-v2-m3");
});

test("desktop: the sidebar Map button opens /map", async ({
  page,
}, testInfo) => {
  test.skip(isMobileProject(testInfo), "the rail is a ≥md surface");
  await primeAppState(page);
  await gotoApp(page);

  await page.getByRole("button", { name: "Map", exact: true }).click();
  await page.waitForURL(/\/map(\?|$)/, { timeout: 15_000 });
  await expect(page.locator(CANVAS)).toBeVisible();
});

test("mobile: the nav drawer's Map entry opens /map", async ({
  page,
}, testInfo) => {
  test.skip(!isMobileProject(testInfo), "the Map entry lives in the mobile drawer");
  await primeAppState(page);
  await gotoApp(page);

  await page.getByRole("button", { name: "Open navigation" }).click();
  const drawer = page.locator("[data-slot=drawer-popup]");
  await expect(drawer).toBeVisible();
  await drawer.getByRole("button", { name: "Map", exact: true }).click();

  await page.waitForURL(/\/map(\?|$)/, { timeout: 15_000 });
  await expect(page.locator(CANVAS)).toBeVisible();
});

test("palette: the Codebase Map command opens /map", async ({ page }) => {
  await primeAppState(page);
  await gotoApp(page);

  await page.keyboard.press("ControlOrMeta+k");
  const palette = page.locator("[data-slot=command-palette]");
  await expect(palette).toBeVisible();

  await page.getByLabel("Type a command or search").fill("codebase");
  const command = palette.getByRole("option", {
    name: "Codebase Map",
    exact: true,
  });
  await expect(command).toBeVisible();
  await command.click();

  await page.waitForURL(/\/map(\?|$)/, { timeout: 15_000 });
  await expect(page.locator(CANVAS)).toBeVisible();
});

test("spotlighting a card dims unrelated cards and shows its detail; Escape clears", async ({
  page,
}) => {
  await primeAppState(page);
  await page.goto("/map");

  const canvas = page.locator(CANVAS);
  await expect(canvas).toBeVisible();

  // The FastAPI backend has a rich downstream closure; the Corpus manager
  // (reached only from the GUI entry) is not in it.
  const source = canvas.getByRole("button", {
    name: new RegExp("^FastAPI backend,"),
  });
  const unrelated = canvas.getByRole("button", {
    name: new RegExp("^Corpus manager,"),
  });

  // Focus first (cards can sit under the floating panel/legend overlays, so
  // focus-then-dispatch avoids a flaky actionability wait while still driving
  // the card's real onClick), then fire the spotlight.
  await source.focus();
  await expect(source).toBeFocused();
  await source.dispatchEvent("click");

  // The detail popover renders the node's `detail` + monospace `sourceRef`.
  await expect(
    page.getByRole("heading", { name: "FastAPI backend" }),
  ).toBeVisible();
  await expect(page.getByText("varagity/api/main.py")).toBeVisible();
  await expect(
    page.getByText(/Async edge over sync Prefect flows/),
  ).toBeVisible();

  // An unrelated card drops to the spotlight-out opacity; Escape restores it.
  await expect(unrelated).toHaveCSS("opacity", "0.25");
  await page.keyboard.press("Escape");
  await expect(
    page.getByRole("heading", { name: "FastAPI backend" }),
  ).toBeHidden();
  await expect(unrelated).toHaveCSS("opacity", "1");
});

test("keyboard: Tab reaches cards in reading order and Enter spotlights", async ({
  page,
}) => {
  await primeAppState(page);
  await page.goto("/map");

  const canvas = page.locator(CANVAS);
  await expect(canvas).toBeVisible();

  // Cards render in declaration (reading) order, so the first two rendered
  // nodes are consecutive in the tab order.
  const first = canvas.getByRole("button", {
    name: new RegExp("^Next\\.js chat GUI,"),
  });
  const second = canvas.getByRole("button", {
    name: new RegExp("^CLI,"),
  });

  await first.focus();
  await expect(first).toBeFocused();

  // Enter activates (spotlights) the focused card — a native button.
  await page.keyboard.press("Enter");
  await expect(
    page.getByRole("heading", { name: "Next.js chat GUI" }),
  ).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(
    page.getByRole("heading", { name: "Next.js chat GUI" }),
  ).toBeHidden();

  // Tab moves focus to the next card.
  await first.focus();
  await page.keyboard.press("Tab");
  await expect(second).toBeFocused();
});

test("developer mode off hides the entry points, but /map still loads (D7)", async ({
  page,
}) => {
  await primeAppState(page, { developerMode: false });
  await gotoApp(page);

  // The sidebar Map button is gone (it renders only when developer mode is on).
  await expect(
    page.getByRole("button", { name: "Map", exact: true }),
  ).toHaveCount(0);

  // …and so is the palette command — search it and confirm zero hits (this
  // covers both palette derivations: a leftover in Base UI's item registry
  // would still surface here).
  await page.keyboard.press("ControlOrMeta+k");
  const palette = page.locator("[data-slot=command-palette]");
  await expect(palette).toBeVisible();
  await page.getByLabel("Type a command or search").fill("codebase");
  await expect(
    palette.getByRole("option", { name: "Codebase Map" }),
  ).toHaveCount(0);
  await page.keyboard.press("Escape");

  // Cosmetic gate only: the route stays reachable by URL.
  await page.goto("/map");
  await expect(page.locator(CANVAS)).toBeVisible();
});

// Axe (critical) across both themes × both densities. The density axis is new
// to this spec — the global a11y.spec sweep is not widened.
for (const theme of THEMES) {
  for (const density of DENSITIES) {
    test(`/map: no critical axe violations (${theme}, ${density})`, async ({
      page,
    }, testInfo) => {
      await primeAppState(page, { theme, density });
      await page.goto("/map");

      const canvas = page.locator(CANVAS);
      await expect(canvas).toBeVisible();
      await expect(canvas.locator("[data-map-node]")).toHaveCount(
        RENDERED_NODES.length,
      );

      await expectNoCriticalA11y(
        page,
        testInfo,
        `${theme}/${density}/${testInfo.project.name}`,
      );
    });
  }
}
