import { expect, test } from "@playwright/test";

import { gotoFreshConversation, isMobileProject, primeAppState } from "./helpers";

// One full LLM generation on the local GPUs runs 30–90 s; give headroom.
const GENERATION_TIMEOUT = 120_000;

// The first preview (locate + page render, and a possible one-time soffice
// conversion for pptx) is server-side work on top of the image fetch.
const PREVIEW_TIMEOUT = 30_000;

// Aimed at the digital-PDF fixture that ships in tests/fixtures/corpus and
// the standing docs/ corpus alike. Retrieval — not the model — decides
// whether a preview-eligible chunk surfaces, and the spec annotates
// instead of failing when none does (other corpora, no digital PDFs).
const QUESTION =
  "What does the Saltmere observatory report describe? Answer briefly.";

test.beforeEach(async ({ page }) => {
  await primeAppState(page);
});

/**
 * The Kotaemon-style page preview (ADR-010) end to end: ask a question
 * that retrieves from a digital PDF, expand an eligible chunk card's
 * "Show preview", and assert the rendered page image, the highlight
 * overlays, and the page footer. Desktop additionally exercises the
 * enlarge dialog (open → Esc closes); the mobile bottom sheet skips the
 * nested-dialog interaction and closes the sheet instead.
 */
test("evidence preview: page image, highlight rects, enlarge dialog", async ({
  page,
}, testInfo) => {
  test.setTimeout(GENERATION_TIMEOUT + 120_000);
  const mobile = isMobileProject(testInfo);

  await gotoFreshConversation(page);

  // ── Ask, and let the turn settle (stable cards, no stream re-renders) ─
  const composer = page.getByLabel("Question");
  await composer.fill(QUESTION);
  await composer.press("Enter");

  const sourcesAffordance = page.getByTitle("Show how this answer was built");
  await expect(sourcesAffordance).toBeVisible({ timeout: GENERATION_TIMEOUT });
  await expect(page.getByRole("button", { name: "Send" })).toBeVisible({
    timeout: GENERATION_TIMEOUT,
  });

  // ── Surface the evidence: right rail (≥lg) or bottom sheet (<lg) ─────
  const evidence = mobile
    ? page.locator("[data-slot=drawer-popup]")
    : page.getByRole("complementary", { name: "How this answer was built" });
  if (mobile) {
    await sourcesAffordance.click();
  }
  await expect(evidence).toBeVisible();
  await expect(evidence.locator("[data-chunk-key]").first()).toBeVisible();

  // ── Find a preview-eligible card (digital pdf / pptx ⇒ "Show preview") ─
  const showPreview = evidence.getByRole("button", { name: "Show preview" });
  if ((await showPreview.count()) === 0) {
    testInfo.annotations.push({
      type: "note",
      description:
        "no preview-eligible (digital pdf/pptx) chunk in this answer's evidence — preview assertions skipped",
    });
    return;
  }
  // Pin the card by its key BEFORE clicking: a filter on "has the Show
  // preview button" would stop matching the moment the label flips to
  // "Hide preview", silently re-resolving to a different card.
  const chunkKey = await evidence
    .locator("[data-chunk-key]")
    .filter({ has: page.getByRole("button", { name: "Show preview" }) })
    .first()
    .getAttribute("data-chunk-key");
  const card = evidence.locator(`[data-chunk-key="${chunkKey}"]`);
  await card.getByRole("button", { name: "Show preview" }).click();
  await expect(
    card.getByRole("button", { name: "Hide preview" }),
  ).toBeVisible();

  // ── The located page renders with the chunk highlighted ──────────────
  const image = card.locator("figure img");
  await expect(image).toBeVisible({ timeout: PREVIEW_TIMEOUT });
  await expect
    .poll(
      () => image.evaluate((el: HTMLImageElement) => el.naturalWidth),
      {
        timeout: PREVIEW_TIMEOUT,
        message: "the rendered page PNG should load (naturalWidth > 0)",
      },
    )
    .toBeGreaterThan(0);
  // At least one highlight overlay rect sits over the image.
  expect(
    await card.locator("figure div.absolute[aria-hidden]").count(),
  ).toBeGreaterThan(0);
  // The "page N of M" footer.
  await expect(card.locator("figcaption")).toContainText(/page \d+ of \d+/);

  if (mobile) {
    // Nested dialog-in-drawer stays a desktop assertion; close the sheet.
    await page.keyboard.press("Escape");
    await expect(evidence).toBeHidden();
    return;
  }

  // ── Enlarge dialog: opens at reading size, Esc closes ────────────────
  await card.getByRole("button", { name: "Enlarge preview" }).click();
  const dialog = page.locator("[data-slot=dialog-content]");
  await expect(dialog).toBeVisible();
  await expect(dialog.getByText(/page \d+ of \d+/)).toBeVisible();
  const dialogImage = dialog.locator("img");
  await expect(dialogImage).toBeVisible();
  await expect
    .poll(
      () => dialogImage.evaluate((el: HTMLImageElement) => el.naturalWidth),
      {
        timeout: PREVIEW_TIMEOUT,
        message: "the enlarged page PNG should load (naturalWidth > 0)",
      },
    )
    .toBeGreaterThan(0);
  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
});
