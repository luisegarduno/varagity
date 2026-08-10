import { expect, test } from "@playwright/test";

import {
  gotoFreshConversation,
  graphSkipReason,
  isMobileProject,
  primeAppState,
} from "./helpers";

/**
 * A graph-targeted chat turn end to end (spec_graphrag §4.2, §4.3):
 * composer selector → `ChatRequest.corpus="graph"` → evidence-before-prose
 * with the graph's own units (entities, relations, cited transcript days)
 * → a grounded, cited answer written by *our* generation path, not the
 * engine's.
 *
 * Skips cleanly on a stack whose graph has not been built — the one corpus
 * a test may not create for itself.
 */

// The graph turn spends a keyword-extraction call plus our streamed answer,
// on the same single llama.cpp slot the document corpus uses.
const GENERATION_TIMEOUT = 120_000;

const QUESTION =
  "Who comes up most often in these messages, and what did they talk about? Answer briefly.";

test.beforeEach(async ({ page }) => {
  await primeAppState(page);
});

test("graph turn: selector, graph evidence, day-grain footer", async ({
  page,
  request,
}, testInfo) => {
  const skip = await graphSkipReason(request, { requireIdle: true });
  test.skip(skip !== null, skip ?? "");
  test.setTimeout(GENERATION_TIMEOUT + 120_000);
  const mobile = isMobileProject(testInfo);

  await gotoFreshConversation(page);

  // ── Point the next question at the graph ────────────────────────────
  const graphOption = page.getByRole("button", { name: "Answer from Messages" });
  await expect(graphOption).toBeEnabled();
  await graphOption.click();
  await expect(graphOption).toHaveAttribute("aria-pressed", "true");
  await expect(
    page.getByRole("button", { name: "Answer from Documents" }),
  ).toHaveAttribute("aria-pressed", "false");

  // ── Ask ─────────────────────────────────────────────────────────────
  const composer = page.getByLabel("Question");
  await composer.fill(QUESTION);
  await composer.press("Enter");

  const sourcesAffordance = page.getByTitle("Show how this answer was built");
  await expect(sourcesAffordance).toBeVisible({ timeout: GENERATION_TIMEOUT });
  await expect(page.getByRole("button", { name: "Send" })).toBeVisible({
    timeout: GENERATION_TIMEOUT,
  });

  // ── The evidence is the graph's, in the graph's units ───────────────
  const evidence = mobile
    ? page.locator("[data-slot=drawer-popup]")
    : page.getByRole("complementary", { name: "How this answer was built" });
  if (mobile) {
    await sourcesAffordance.click();
  }
  await expect(evidence).toBeVisible();

  // The corpus badge is on the turn either way; which one it is says
  // whether the graph answered or the turn degraded to the document
  // corpus (ADR-017 — a degrade is reported, never hidden).
  const degraded = evidence.getByText("graph → documents", { exact: true });
  if ((await degraded.count()) > 0) {
    testInfo.annotations.push({
      type: "note",
      description:
        "the graph turn degraded to the document corpus — the badge says so; graph-evidence assertions skipped",
    });
    return;
  }
  await expect(evidence.getByText("graph", { exact: true }).first()).toBeVisible();

  // Cited transcript days, entities, relations — at least the days, which
  // are what the answer is allowed to quote.
  const days = evidence.locator("[data-chunk-key]");
  await expect(days.first()).toBeVisible();
  await expect(
    evidence.getByRole("region", { name: "Transcript days" }),
  ).toBeVisible();
  await expect(evidence.getByText(/\d+ entities · \d+ relations/)).toBeVisible();

  // A day card names its thread and opens the transcript it cited.
  const firstDay = days.first();
  await firstDay.getByRole("button", { name: "Show transcript" }).click();
  await expect(
    firstDay.getByRole("button", { name: "Hide transcript" }),
  ).toBeVisible();

  // The honesty line that makes day-grain evidence readable as such.
  await expect(
    evidence.getByText(/Graph evidence is day-grain/),
  ).toBeVisible();

  // ── Citation chips (model-dependent, as in chat-flow) ───────────────
  const matchedChips = page.locator('[data-citation="matched"]');
  if ((await matchedChips.count()) > 0) {
    await matchedChips.first().click();
    await page.waitForFunction(
      () => document.querySelector(".evidence-pulse") !== null,
      undefined,
      { timeout: 15_000 },
    );
  } else {
    testInfo.annotations.push({
      type: "note",
      description:
        "no matched [SOURCE] chip in this graph answer (model nondeterminism) — citation-pulse assertion skipped",
    });
  }

  // ── The selector stays where the conversation left it ───────────────
  // Derived from the transcript, never persisted separately (decision #20).
  await page.reload();
  await expect(
    page.getByRole("button", { name: "Answer from Messages" }),
  ).toHaveAttribute("aria-pressed", "true");
});
