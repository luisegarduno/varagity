import { expect, test } from "@playwright/test";

import { primeAppState } from "./helpers";

/**
 * The Corpus page's two tabs (spec_graphrag §4.4).
 *
 * Two jobs. The first is a **regression guard**: the document corpus was
 * byte-identical before and after the graph tab arrived, so everything that
 * was on `/corpus` must still be exactly one click away. The second is that
 * the Graph RAG tab renders honestly against whatever the live stack holds
 * — an empty graph, a built one, or a disabled subsystem are all valid, and
 * all three must produce a readable panel rather than a spinner.
 *
 * Strictly read-only toward both corpora: no upload, no ingest, and above
 * all **no build** — pressing Build here would start a multi-day extraction
 * on the owner's real archive.
 */

test.beforeEach(async ({ page }) => {
  await primeAppState(page);
});

test("corpus: Documents is the default tab and still renders the RAG surfaces", async ({
  page,
}) => {
  await page.goto("/corpus");

  await expect(page.getByRole("heading", { name: "Corpus" })).toBeVisible();
  const documents = page.getByRole("tab", { name: "Documents" });
  await expect(documents).toHaveAttribute("aria-selected", "true");

  // The pre-graph page, unchanged: dropzone, ingest panel, document table
  // (or the guided first-run card, on a corpus with nothing ingested).
  const dropzone = page.getByRole("region", { name: "Upload documents" });
  const guided = page.getByRole("region", { name: "Getting started" });
  await expect(dropzone.or(guided).first()).toBeVisible();
  await expect(
    page.getByRole("region", { name: "Ingestion" }).first(),
  ).toBeVisible();
});

test("corpus: the Message graph tab shows the graph's own state", async ({
  page,
}) => {
  await page.goto("/corpus");

  await page.getByRole("tab", { name: "Message graph" }).click();
  await expect(
    page.getByRole("tab", { name: "Message graph" }),
  ).toHaveAttribute("aria-selected", "true");

  // Panels are unmounted while hidden, so the document surfaces go away
  // entirely — that is what keeps a hidden tab from polling.
  await expect(page.getByRole("region", { name: "Ingestion" })).toHaveCount(0);

  // Either the graph corpus panel or the honest "subsystem is off" card.
  const size = page.getByLabel("Graph size");
  const disabled = page.getByRole("region", { name: "Graph disabled" });
  await expect(size.or(disabled).first()).toBeVisible();

  if ((await disabled.count()) > 0) return; // GRAPH_ENABLED=false: nothing more to show

  // The size strip names its three figures whether or not they have values
  // ("—" is the unbuilt reading, deliberately not 0).
  for (const figure of ["entities", "relations", "messages", "thread-days"]) {
    await expect(size.getByText(figure, { exact: true })).toBeVisible();
  }
  // The build panel is present and idle. Never clicked: a build is hours.
  await expect(page.getByRole("region", { name: "Graph build" })).toBeVisible();
  await expect(page.getByRole("region", { name: "Graph sources" })).toBeVisible();
});

test("corpus: ?tab=graph deep-links the graph tab, and Documents comes back", async ({
  page,
}) => {
  // The ⌘K palette's "Open graph corpus" target — resolved server-side, so
  // a cold load must land on the second tab without a client-side flip.
  await page.goto("/corpus?tab=graph");

  await expect(
    page.getByRole("tab", { name: "Message graph" }),
  ).toHaveAttribute("aria-selected", "true");

  await page.getByRole("tab", { name: "Documents" }).click();
  await expect(
    page.getByRole("region", { name: "Ingestion" }).first(),
  ).toBeVisible();
  await expect(page.getByLabel("Graph size")).toHaveCount(0);
});
