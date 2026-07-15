import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page, type TestInfo } from "@playwright/test";

import {
  expectTheme,
  gotoApp,
  gotoFreshConversation,
  isMobileProject,
  primeAppState,
  type ThemeName,
} from "./helpers";

const THEMES: readonly ThemeName[] = ["light", "dark"];

// Scan the settled design, not a transition frame: axe otherwise races the
// entry animations (the hero's 300ms fade-in, the palette popup's 200ms
// scale/opacity) and reads text mid-fade — e.g. the statically-AA hero
// subtitle (6.09:1) measured 4.32:1 at container opacity 0.85. The app
// honors `motion-safe:`/`motion-reduce:` throughout, so reduced-motion is a
// first-class rendering with identical resting colors.
test.use({ contextOptions: { reducedMotion: "reduce" } });

/**
 * Scan the page's current state with axe and gate on CRITICAL violations
 * (the Phase 9 criterion). Serious/moderate counts are logged per state
 * for the report, and the full violation JSON is attached whenever any
 * violation (of any impact) fires.
 */
async function checkA11y(
  page: Page,
  testInfo: TestInfo,
  label: string,
): Promise<void> {
  const results = await new AxeBuilder({ page }).analyze();
  const byImpact = (impact: string) =>
    results.violations.filter((violation) => violation.impact === impact);
  const critical = byImpact("critical");
  const serious = byImpact("serious");
  const moderate = byImpact("moderate");
  console.log(
    `[axe] ${label}: critical=${critical.length} serious=${serious.length} moderate=${moderate.length}` +
      (results.violations.length > 0
        ? ` — ${results.violations
            .map((violation) => `${violation.id}(${violation.impact})`)
            .join(", ")}`
        : ""),
  );
  if (results.violations.length > 0) {
    await testInfo.attach(`axe-${label.replaceAll("/", "-")}`, {
      body: JSON.stringify(results.violations, null, 2),
      contentType: "application/json",
    });
  }
  expect(
    critical,
    `critical axe violations on ${label}:\n${JSON.stringify(critical, null, 2)}`,
  ).toEqual([]);
}

for (const theme of THEMES) {
  test.describe(`${theme} theme`, () => {
    test.beforeEach(async ({ page }) => {
      await primeAppState(page, { theme });
    });

    test(`conversation empty state: no critical violations (${theme})`, async ({
      page,
    }, testInfo) => {
      await gotoFreshConversation(page);
      await expectTheme(page, theme);
      // The pre-first-question hero (the corpus is non-empty ⇒ plain hero).
      await expect(
        page.getByRole("heading", { name: "Ask your corpus" }),
      ).toBeVisible();
      await checkA11y(
        page,
        testInfo,
        `empty-conversation/${theme}/${testInfo.project.name}`,
      );
    });

    test(`corpus page: no critical violations (${theme})`, async ({
      page,
    }, testInfo) => {
      await page.goto("/corpus");
      await expectTheme(page, theme);
      await expect(
        page.getByRole("heading", { name: "Corpus", level: 1 }),
      ).toBeVisible();
      await expect(
        page.getByLabel("Upload documents into the corpus"),
      ).toBeVisible();
      // Scan real content: wait for the ingested-documents rows to land.
      await expect(
        page
          .getByRole("region", { name: "Ingested documents" })
          .locator("tbody tr")
          .first(),
      ).toBeVisible();
      await checkA11y(page, testInfo, `corpus/${theme}/${testInfo.project.name}`);
    });

    test(`settings drawer open: no critical violations (${theme})`, async ({
      page,
    }, testInfo) => {
      test.skip(
        isMobileProject(testInfo),
        "opened via the desktop rail's Settings button",
      );
      await gotoApp(page);
      await expectTheme(page, theme);
      await page.getByRole("button", { name: "Settings" }).click();
      const sheet = page.locator("[data-slot=dialog-content]");
      await expect(sheet).toBeVisible();
      // The form hydrates from the API — scan once real controls landed.
      await expect(sheet.getByRole("button", { name: "Apply" })).toBeVisible();
      await checkA11y(
        page,
        testInfo,
        `settings-drawer/${theme}/${testInfo.project.name}`,
      );
    });

    test(`command palette open: no critical violations (${theme})`, async ({
      page,
    }, testInfo) => {
      await gotoApp(page);
      await expectTheme(page, theme);
      await page.keyboard.press("ControlOrMeta+k");
      await expect(page.locator("[data-slot=command-palette]")).toBeVisible();
      await expect(page.getByLabel("Type a command or search")).toBeFocused();
      await checkA11y(
        page,
        testInfo,
        `command-palette/${theme}/${testInfo.project.name}`,
      );
    });
  });
}
