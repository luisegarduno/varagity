import { describe, expect, it } from "vitest";

import type { ConversationSummary, GroupOut } from "@/lib/api";
import {
  activeGroupId,
  CONVERSATION_DRAG_TYPE,
  conversationIdFromPathname,
  groupConversations,
  isGroupExpanded,
} from "@/lib/conversation-groups";

function conversation(
  id: string,
  groupId: string | null = null,
): ConversationSummary {
  return {
    conversation_id: id,
    title: `Chat ${id}`,
    created_at: "2026-07-22T00:00:00Z",
    updated_at: "2026-07-22T00:00:00Z",
    message_count: 0,
    group_id: groupId,
  };
}

function group(id: string, name = `Group ${id}`): GroupOut {
  return { group_id: id, name, created_at: "2026-07-22T00:00:00Z" };
}

describe("groupConversations", () => {
  it("partitions by group, preserving both input orders", () => {
    const groups = [group("g1"), group("g2")];
    const conversations = [
      conversation("a", "g2"),
      conversation("b"),
      conversation("c", "g1"),
      conversation("d", "g2"),
    ];

    const { sections, ungrouped } = groupConversations(groups, conversations);

    // Sections follow the group list's order…
    expect(sections.map((section) => section.group.group_id)).toEqual(["g1", "g2"]);
    // …and each keeps the conversation list's (recency) order.
    expect(sections[1].conversations.map((c) => c.conversation_id)).toEqual([
      "a",
      "d",
    ]);
    expect(ungrouped.map((c) => c.conversation_id)).toEqual(["b"]);
  });

  it("keeps empty groups as sections — an empty folder is a drop target", () => {
    const { sections } = groupConversations([group("g1")], []);
    expect(sections).toHaveLength(1);
    expect(sections[0].conversations).toEqual([]);
  });

  it("degrades a conversation pointing at an unknown group to ungrouped", () => {
    // A stale cache mid-delete: the group vanished before the list refetched.
    const { sections, ungrouped } = groupConversations(
      [group("g1")],
      [conversation("a", "ghost")],
    );
    expect(sections[0].conversations).toEqual([]);
    expect(ungrouped.map((c) => c.conversation_id)).toEqual(["a"]);
  });

  it("handles the empty world", () => {
    expect(groupConversations([], [])).toEqual({ sections: [], ungrouped: [] });
  });
});

describe("conversationIdFromPathname", () => {
  it("extracts the id from a conversation route", () => {
    expect(conversationIdFromPathname("/c/abc123")).toBe("abc123");
  });

  it("decodes percent-encoded ids", () => {
    expect(conversationIdFromPathname("/c/a%20b")).toBe("a b");
  });

  it("returns null off the conversation route", () => {
    expect(conversationIdFromPathname("/")).toBeNull();
    expect(conversationIdFromPathname("/corpus")).toBeNull();
    expect(conversationIdFromPathname("/c/")).toBeNull();
    expect(conversationIdFromPathname("/c/a/b")).toBeNull();
  });
});

describe("activeGroupId", () => {
  const sections = groupConversations(
    [group("g1"), group("g2")],
    [conversation("a", "g1"), conversation("b")],
  ).sections;

  it("names the group holding the active conversation", () => {
    expect(activeGroupId(sections, "a")).toBe("g1");
  });

  it("returns null for ungrouped, unknown, or no active conversation", () => {
    expect(activeGroupId(sections, "b")).toBeNull();
    expect(activeGroupId(sections, "ghost")).toBeNull();
    expect(activeGroupId(sections, null)).toBeNull();
  });
});

describe("isGroupExpanded", () => {
  it("collapses by default", () => {
    expect(isGroupExpanded("g1", {}, null)).toBe(false);
  });

  it("auto-expands the group holding the active conversation", () => {
    expect(isGroupExpanded("g1", {}, "g1")).toBe(true);
    expect(isGroupExpanded("g2", {}, "g1")).toBe(false);
  });

  it("lets an explicit toggle beat the active-group default", () => {
    // Collapsed by hand while active inside — stays collapsed…
    expect(isGroupExpanded("g1", { g1: false }, "g1")).toBe(false);
    // …and expanded by hand stays open after navigating away.
    expect(isGroupExpanded("g1", { g1: true }, null)).toBe(true);
  });
});

describe("CONVERSATION_DRAG_TYPE", () => {
  it("is a custom MIME type no other drag source produces", () => {
    expect(CONVERSATION_DRAG_TYPE).toBe("application/x-varagity-conversation");
  });
});
