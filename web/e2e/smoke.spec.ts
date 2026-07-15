import { expect, test } from "@playwright/test";

import {
  CONVERSATION_URL,
  gotoApp,
  isMobileProject,
  primeAppState,
} from "./helpers";

test.beforeEach(async ({ page }) => {
  await primeAppState(page);
});

test("/ redirects to a conversation with the composer ready", async ({
  page,
}) => {
  await gotoApp(page);
  expect(page.url()).toMatch(CONVERSATION_URL);
  await expect(page.getByLabel("Question")).toBeVisible();
  // The composer's kbd hint row is aria-hidden — assert via the slot.
  await expect(page.locator("[data-slot=kbd]").first()).toBeAttached();
});

test("desktop: the sidebar rail is visible", async ({ page }, testInfo) => {
  test.skip(isMobileProject(testInfo), "the rail is a ≥md surface");
  await gotoApp(page);
  await expect(
    page.getByRole("navigation", { name: "Conversations" }),
  ).toBeVisible();
  await expect(page.getByRole("button", { name: "New chat" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Settings" })).toBeVisible();
  // The mobile top bar stays hidden at desktop width.
  await expect(
    page.getByRole("button", { name: "Open navigation" }),
  ).toBeHidden();
});

test("mobile: hamburger opens the navigation drawer, Esc closes it", async ({
  page,
}, testInfo) => {
  test.skip(!isMobileProject(testInfo), "the drawer is a <md surface");
  await gotoApp(page);
  await page.getByRole("button", { name: "Open navigation" }).click();
  const drawer = page.locator("[data-slot=drawer-popup]");
  await expect(drawer).toBeVisible();
  // The drawer hosts the same navigation content as the desktop rail.
  await expect(
    drawer.getByRole("navigation", { name: "Conversations" }),
  ).toBeVisible();
  await expect(drawer.getByRole("button", { name: "New chat" })).toBeVisible();
  await page.keyboard.press("Escape");
  // Drawer popups unmount once the exit transition settles.
  await expect(drawer).toBeHidden();
});

test("skip link is the first focusable and targets #main", async ({ page }) => {
  await page.goto("/corpus");
  await expect(page.getByRole("heading", { name: "Corpus" })).toBeVisible();
  // Neutralize any autofocus so Tab starts from the top of the document.
  await page.evaluate(() => {
    (document.activeElement as HTMLElement | null)?.blur?.();
  });
  await page.keyboard.press("Tab");
  const focused = page.locator(":focus");
  await expect(focused).toHaveAttribute("href", "#main");
  await expect(focused).toHaveText("Skip to content");
  await expect(page.locator("main#main")).toBeAttached();
});

test("/corpus renders the corpus management surface", async ({ page }) => {
  await page.goto("/corpus");
  await expect(page.getByRole("heading", { name: "Corpus" })).toBeVisible();
  await expect(
    page.getByLabel("Upload documents into the corpus"),
  ).toBeVisible();
  // The live corpus is non-empty: the ingested-documents section renders.
  await expect(
    page.getByRole("region", { name: "Ingested documents" }),
  ).toBeVisible();
});
