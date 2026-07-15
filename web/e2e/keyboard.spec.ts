import { expect, test } from "@playwright/test";

import { gotoApp, isMobileProject, primeAppState } from "./helpers";

test.beforeEach(async ({ page }) => {
  await primeAppState(page);
});

test("palette: ⌘K opens, typing filters, Esc closes without navigating", async ({
  page,
}) => {
  await gotoApp(page);
  const urlBefore = page.url();

  await page.keyboard.press("ControlOrMeta+k");
  const palette = page.locator("[data-slot=command-palette]");
  await expect(palette).toBeVisible();
  const input = page.getByLabel("Type a command or search");
  await expect(input).toBeFocused();

  // The un-queried inventory shows the static commands…
  await expect(palette.getByText("Focus composer")).toBeVisible();
  // …and typing filters the list down.
  await input.fill("theme");
  await expect(palette.getByText("Theme: Dark")).toBeVisible();
  await expect(palette.getByText("Focus composer")).toBeHidden();

  await page.keyboard.press("Escape");
  await expect(palette).toBeHidden();
  expect(page.url()).toBe(urlBefore); // closed without navigating
});

test("palette: running the Corpus command navigates to /corpus", async ({
  page,
}) => {
  await gotoApp(page);
  await page.keyboard.press("ControlOrMeta+k");
  const palette = page.locator("[data-slot=command-palette]");
  await expect(palette).toBeVisible();

  const input = page.getByLabel("Type a command or search");
  await input.fill("corpus");
  // Label-prefix matches outrank conversation-title hits and static
  // commands precede conversations at equal rank (lib/palette.ts), so the
  // highlighted first item is deterministically the Corpus command.
  const corpusOption = palette.getByRole("option", {
    name: "Corpus",
    exact: true,
  });
  await expect(corpusOption).toBeVisible();
  await page.keyboard.press("Enter");

  await page.waitForURL(/\/corpus(\?|$)/, { timeout: 15_000 });
  await expect(
    page.getByRole("heading", { name: "Corpus", level: 1 }),
  ).toBeVisible();
});

test("settings sheet: opens from the sidebar, Esc closes it and nothing else", async ({
  page,
}, testInfo) => {
  test.skip(
    isMobileProject(testInfo),
    "the Settings affordance under test lives in the desktop rail",
  );
  await gotoApp(page);
  const urlBefore = page.url();

  await page.getByRole("button", { name: "Settings" }).click();
  const sheet = page.locator("[data-slot=dialog-content]");
  await expect(sheet).toBeVisible();
  await expect(sheet.getByText("Query-time knobs")).toBeVisible();

  await page.keyboard.press("Escape");
  await expect(sheet).toBeHidden();
  expect(page.url()).toBe(urlBefore); // Esc closed the sheet, nothing more
  await expect(page.getByLabel("Question")).toBeVisible();
});
