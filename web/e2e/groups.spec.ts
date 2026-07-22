import { expect, test } from "@playwright/test";

import { gotoApp, isMobileProject, primeAppState } from "./helpers";

const API = "http://localhost:8000";
const STAMP = Date.now();
const GROUP = `e2e-group-${STAMP}`;
const CHAT = `e2e-chat-${STAMP}`;
// The header button's accessible name is "<name> <member count>" — anchored
// so it can never collide with the "Delete group <name>" button's name.
const GROUP_HEADER_RE = new RegExp(
  `^${GROUP.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\s+\\d+$`,
);

test.describe("sidebar conversation groups", () => {
  test.afterAll(async ({ request }) => {
    // Leave the live DB as we found it.
    const conversations = await (await request.get(`${API}/api/conversations`)).json();
    for (const c of conversations) {
      if (c.title === CHAT) {
        await request.delete(`${API}/api/conversations/${c.conversation_id}`);
      }
    }
    const groups = await (await request.get(`${API}/api/groups`)).json();
    for (const g of groups) {
      if (g.name === GROUP) await request.delete(`${API}/api/groups/${g.group_id}`);
    }
  });

  test("create, drag in, collapse rules, ungroup, delete", async ({ page }, testInfo) => {
    test.skip(isMobileProject(testInfo), "drag-and-drop and hover affordances are ≥md surfaces");
    await primeAppState(page, { theme: "light" });
    await gotoApp(page);

    const sidebar = page.getByRole("navigation", { name: "Conversations" });
    const groupHeader = sidebar.getByRole("button", { name: GROUP_HEADER_RE });

    // ── 1. Split header: New chat | New group side by side.
    await expect(page.getByRole("button", { name: "New chat" })).toBeVisible();
    await page.getByRole("button", { name: "New group" }).click();
    await page.getByLabel("Group name").fill(GROUP);
    await page.getByRole("button", { name: "Create", exact: true }).click();

    // The new (empty) group opens itself and shows its drop affordance.
    await expect(groupHeader).toBeVisible();
    await expect(groupHeader).toHaveAttribute("aria-expanded", "true");
    await expect(sidebar.getByText("Drag chats here")).toBeVisible();

    // ── 2. A fresh conversation (seeded via API for a unique title).
    const seeded = await page.request.post(`${API}/api/conversations`, {
      data: { title: CHAT },
    });
    const chatId = (await seeded.json()).conversation_id as string;
    await page.reload();
    const chatRow = sidebar.getByRole("button", { name: CHAT, exact: true });
    await expect(chatRow).toBeVisible();

    // ── 3. Drag it onto the group (HTML5 DnD). The reload above reset the
    // in-memory expand override, so the group sits collapsed — the count
    // ticking to 1 is the proof the drop landed; expanding shows the row.
    await chatRow.locator("xpath=ancestor::li[1]").dragTo(groupHeader);
    await expect(groupHeader).toContainText("1"); // member count
    await groupHeader.click();
    const groupSection = sidebar.locator("li", {
      has: page.getByRole("button", { name: GROUP_HEADER_RE }),
    });
    await expect(groupSection.getByRole("button", { name: CHAT, exact: true })).toBeVisible();

    // ── 4. Collapsed by default on a fresh load (chat inside is not active).
    await page.reload();
    await expect(groupHeader).toHaveAttribute("aria-expanded", "false");
    await expect(sidebar.getByRole("button", { name: CHAT, exact: true })).toBeHidden();

    // Manual expand shows the filed chat.
    await groupHeader.click();
    await expect(groupHeader).toHaveAttribute("aria-expanded", "true");
    const filedRow = sidebar.getByRole("button", { name: CHAT, exact: true });
    await expect(filedRow).toBeVisible();

    // ── 5. Open the filed chat: its group auto-expands on a fresh load.
    await filedRow.click();
    await page.waitForURL(new RegExp(`/c/${chatId}`));
    await page.reload();
    await expect(groupHeader).toHaveAttribute("aria-expanded", "true");
    await expect(sidebar.getByRole("button", { name: CHAT, exact: true })).toBeVisible();

    // An explicit collapse beats the active-conversation default.
    await groupHeader.click();
    await expect(groupHeader).toHaveAttribute("aria-expanded", "false");
    await expect(sidebar.getByRole("button", { name: CHAT, exact: true })).toBeHidden();

    // ── 6. The ⋯ menu is the pointer-free path: remove from group.
    await groupHeader.click(); // re-expand to reach the row
    await sidebar.getByRole("button", { name: `Move ${CHAT} to a group` }).click();
    await page.getByRole("menuitem", { name: "Remove from group" }).click();
    await expect(groupSection.getByRole("button", { name: CHAT, exact: true })).toBeHidden();
    await expect(sidebar.getByRole("button", { name: CHAT, exact: true })).toBeVisible();

    // ── 7. Delete the group; the sidebar returns to a flat list.
    await sidebar.getByRole("button", { name: `Delete group ${GROUP}` }).click();
    await page.getByRole("button", { name: "Delete", exact: true }).click();
    await expect(groupHeader).toBeHidden();
    await expect(sidebar.getByRole("button", { name: CHAT, exact: true })).toBeVisible();
  });
});
