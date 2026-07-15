import { expect, test } from "@playwright/test";

import { gotoFreshConversation, isMobileProject, primeAppState } from "./helpers";

// One full LLM generation on the local GPUs runs 30–90 s; give headroom.
const GENERATION_TIMEOUT = 120_000;

const QUESTION =
  "What topics do the documents in this corpus cover? Answer briefly.";
const FOLLOW_UP = "And what file formats do they use? Answer briefly.";

test.beforeEach(async ({ page }) => {
  await primeAppState(page);
});

/**
 * The full ask journey in one pass (LLM turns are expensive — one real
 * generation per project, plus a second question that Esc stops early):
 * ask → stage indicator → streamed answer → evidence (rail or sheet) →
 * citation pulse (model-permitting) → Esc-to-stop.
 */
test("full ask flow: stages, streamed answer, evidence, Esc stops", async ({
  page,
}, testInfo) => {
  test.setTimeout(GENERATION_TIMEOUT * 2 + 120_000);
  const mobile = isMobileProject(testInfo);

  await gotoFreshConversation(page);

  // ── Ask ─────────────────────────────────────────────────────────────
  const composer = page.getByLabel("Question");
  await composer.fill(QUESTION);
  await composer.press("Enter");

  // The optimistic user bubble lands immediately.
  await expect(page.getByText(QUESTION).first()).toBeVisible();

  // The stage indicator marks the live turn — the active stage shimmers,
  // starting at "Retrieving" — and the send button flips to Stop.
  await expect(page.locator(".shimmer").first()).toBeVisible();
  // exact: the sr-only live region says "Retrieving…" (ellipsis) alongside.
  await expect(page.getByText("Retrieving", { exact: true })).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Stop generating" }),
  ).toBeVisible();

  // ── The answer streams in ───────────────────────────────────────────
  // Evidence-before-prose: the per-answer sources affordance mounts once
  // the retrieval event landed AND answer tokens started rendering.
  const sourcesAffordance = page.getByTitle("Show how this answer was built");
  await expect(sourcesAffordance).toBeVisible({ timeout: GENERATION_TIMEOUT });

  // Non-trivial prose accumulates in the assistant bubble (mid-stream or
  // already settled — either proves the answer arrived).
  const answerProse = page
    .locator("[data-slot=bubble]")
    .last()
    .locator(".prose-chat")
    .last();
  await expect
    .poll(async () => ((await answerProse.textContent()) ?? "").trim().length, {
      timeout: GENERATION_TIMEOUT,
      message: "answer prose should grow past a trivial length",
    })
    .toBeGreaterThan(20);

  // The turn settles: the composer returns to Send and the indicator is
  // gone (it yields to the evidence panel's numbers).
  await expect(page.getByRole("button", { name: "Send" })).toBeVisible({
    timeout: GENERATION_TIMEOUT,
  });
  await expect(page.locator(".shimmer")).toHaveCount(0);
  await expect(page.getByText("Retrieving")).toHaveCount(0);

  // ── Evidence: “how this answer was built” ───────────────────────────
  if (!mobile) {
    // ≥lg: the right rail (pinned open by primeAppState) populates.
    const rail = page.getByRole("complementary", {
      name: "How this answer was built",
    });
    await expect(rail).toBeVisible();
    const cards = rail.locator("[data-chunk-key]");
    await expect(cards.first()).toBeVisible();
    expect(await cards.count()).toBeGreaterThanOrEqual(1);
    // The answer-level retrieval-method badge, and per-chunk trace badges.
    await expect(rail.locator("[data-slot=badge]").first()).toBeVisible();
    await expect(rail.locator("[data-kind]").first()).toBeVisible();
  } else {
    // <lg: the per-message sources affordance opens the bottom sheet.
    await sourcesAffordance.click();
    const sheet = page.locator("[data-slot=drawer-popup]");
    await expect(sheet).toBeVisible();
    await expect(sheet.getByText("How this answer was built")).toBeVisible();
    await expect(sheet.locator("[data-chunk-key]").first()).toBeVisible();
    await expect(sheet.locator("[data-slot=badge]").first()).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(sheet).toBeHidden(); // unmounts after the exit transition
  }

  // ── Citation pulse (model-dependent: tolerate absent chips) ──────────
  const matchedChips = page.locator('[data-citation="matched"]');
  if ((await matchedChips.count()) > 0) {
    await matchedChips.first().click();
    // The cited chunk card scrolls into view and pulses (class removed on
    // animationend, so poll rather than assert a steady state).
    await page.waitForFunction(
      () => document.querySelector(".evidence-pulse") !== null,
      undefined,
      { timeout: 15_000 },
    );
    if (mobile) {
      // The chip reopened the bottom sheet — close it so Escape below
      // reaches the stream, not the sheet.
      await page.keyboard.press("Escape");
      await expect(page.locator("[data-slot=drawer-popup]")).toBeHidden();
    }
  } else {
    testInfo.annotations.push({
      type: "note",
      description:
        "no matched [SOURCE] chip in this answer (model nondeterminism) — citation-pulse assertion skipped",
    });
  }

  // ── Esc stops the second question’s stream ──────────────────────────
  await composer.fill(FOLLOW_UP);
  await composer.press("Enter");
  await expect(page.locator(".shimmer").first()).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Stop generating" }),
  ).toBeVisible();
  await page.keyboard.press("Escape");

  // The composer returns to the Send state far sooner than a full
  // generation ever could — that is the point of Esc-to-stop.
  await expect(page.getByRole("button", { name: "Send" })).toBeVisible({
    timeout: 15_000,
  });
  const stoppedNotice = page.getByText(/Stopped — this partial answer/);
  if ((await stoppedNotice.count()) > 0) {
    await expect(stoppedNotice.first()).toBeVisible();
  } else {
    // Only possible if the turn settled in the instant before Escape
    // landed — tolerate the nondeterminism instead of flaking.
    testInfo.annotations.push({
      type: "note",
      description:
        "turn settled before Escape landed — the stop path was not exercised this run",
    });
  }
});
