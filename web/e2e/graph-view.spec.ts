import { expect, test, type APIRequestContext } from "@playwright/test";

import {
  API_URL,
  gotoApp,
  graphSkipReason,
  isMobileProject,
  primeAppState,
} from "./helpers";

/**
 * `/graph` — the whole extracted graph, drawn offline (spec_graphrag §4.4):
 * it renders, it is searchable, a node opens its source days, and the
 * export cap announces itself rather than implying it drew everything.
 *
 * Skips cleanly on a stack whose graph has not been built. Assertions stay
 * DOM-level (the canvas region, the panels around it) rather than reaching
 * into WebGL pixels — sigma draws nodes as pixels, which is exactly why the
 * search box is the keyboard path *to* a node.
 */

/** The canvas region — `aria-label` on the `<section>` wrapping sigma. */
const CANVAS = 'section[aria-label="Message graph canvas"]';

/** The most connected entities, straight from the export the view draws. */
async function topEntities(
  request: APIRequestContext,
  maxNodes: number,
): Promise<{ nodes: { id: string }[]; truncated: boolean }> {
  const response = await request.get(
    `${API_URL}/api/graph/export?max_nodes=${maxNodes}`,
  );
  expect(response.ok(), "GET /api/graph/export should answer").toBeTruthy();
  return response.json();
}

test.beforeEach(async ({ page }) => {
  await primeAppState(page);
});

test("graph view: renders the canvas, the legend, and the entity count", async ({
  page,
  request,
}, testInfo) => {
  const skip = await graphSkipReason(request);
  test.skip(skip !== null, skip ?? "");

  await page.goto("/graph");

  await expect(
    page.getByRole("heading", { name: "Message graph" }),
  ).toBeVisible();
  await expect(page.locator(CANVAS)).toBeVisible({ timeout: 30_000 });

  // The sr-only description is the accessible account of what was drawn.
  await expect(
    page.getByText(/Interactive graph of \d+ entities and \d+ relations/),
  ).toBeAttached();

  if (!isMobileProject(testInfo)) {
    // The legend groups entity *types* — LightRAG has no communities, and
    // the copy says so rather than implying clusters (ADR-017). Exact: the
    // page description also contains the phrase mid-sentence.
    await expect(page.getByText("entity type", { exact: true })).toBeVisible();
  }
});

test("graph view: search centres a match and opens its source days", async ({
  page,
  request,
}) => {
  const skip = await graphSkipReason(request);
  test.skip(skip !== null, skip ?? "");

  const { nodes } = await topEntities(request, 25);
  test.skip(nodes.length === 0, "the export returned no entities");
  const name = nodes[0].id;
  // Search matches on the label prefix; a short slice keeps the needle
  // clear of any punctuation an extracted name may carry.
  const needle = name.slice(0, Math.min(4, name.length));

  await page.goto("/graph");
  await expect(page.locator(CANVAS)).toBeVisible({ timeout: 30_000 });

  const search = page.getByLabel("Search entities");
  await search.fill(needle);
  const results = page.getByRole("list", { name: "Search results" });
  await expect(results).toBeVisible();
  await expect(results.getByRole("button").first()).toBeVisible();

  // Enter centres the first match and pins it — the inspect panel opens on
  // the entity's own name, so a WebGL node becomes a readable record.
  await search.press("Enter");
  const panel = page.getByRole("complementary", { name: `Entity ${name}` });
  await expect(panel).toBeVisible();
  await expect(panel.getByRole("heading", { name, exact: true })).toBeVisible();
  await expect(
    panel.getByRole("heading", { name: "Relations", exact: true }),
  ).toBeVisible();
  await expect(
    panel.getByRole("heading", { name: "Source days", exact: true }),
  ).toBeVisible();
  await expect(panel.getByText(/Evidence is day-grain/)).toBeVisible();

  // Escape clears the selection (focus was handed to the canvas region, so
  // the key lands inside the view rather than on <body>).
  await page.keyboard.press("Escape");
  await expect(panel).toBeHidden();
});

test("graph view: the export cap announces itself", async ({
  page,
  request,
}, testInfo) => {
  const skip = await graphSkipReason(request);
  test.skip(skip !== null, skip ?? "");

  const { truncated } = await topEntities(request, 2);
  if (!truncated) {
    testInfo.annotations.push({
      type: "note",
      description:
        "the live graph fits inside 2 nodes — the truncation notice cannot be exercised",
    });
    return;
  }

  await page.goto("/graph?max_nodes=2");
  await expect(page.locator(CANVAS)).toBeVisible({ timeout: 30_000 });
  await expect(
    page.getByRole("status").filter({ hasText: /most connected entities/ }),
  ).toBeVisible();
});

test("desktop: the sidebar Graph entry opens /graph", async ({
  page,
}, testInfo) => {
  test.skip(isMobileProject(testInfo), "the rail is a ≥md surface");
  // Ungated, unlike Map: the graph is a product surface, not a developer
  // one, so it renders with developer mode explicitly off.
  await primeAppState(page, { developerMode: false });
  await gotoApp(page);

  await page.getByRole("button", { name: "Graph", exact: true }).click();
  await page.waitForURL(/\/graph(\?|$)/, { timeout: 15_000 });
  await expect(
    page.getByRole("heading", { name: "Message graph" }),
  ).toBeVisible();
});

test("palette: the graph view command opens /graph", async ({ page }) => {
  await gotoApp(page);

  await page.keyboard.press("ControlOrMeta+k");
  const palette = page.locator("[data-slot=command-palette]");
  await expect(palette).toBeVisible();

  await page.getByLabel("Type a command or search").fill("graph view");
  const command = palette.getByRole("option", {
    name: "Open graph view",
    exact: true,
  });
  await expect(command).toBeVisible();
  await command.click();

  await page.waitForURL(/\/graph(\?|$)/, { timeout: 15_000 });
  await expect(
    page.getByRole("heading", { name: "Message graph" }),
  ).toBeVisible();
});
