/**
 * Pure logic for the sidebar's conversation groups: partitioning the
 * conversation list into group sections, deriving which group holds the
 * active conversation, and resolving each group's expanded state. Kept out
 * of the component so the rules are unit-testable
 * (`lib/__tests__/conversation-groups.test.ts`).
 */
import type { ConversationSummary, GroupOut } from "@/lib/api";

/**
 * The drag payload type marking a sidebar conversation row. Drop targets
 * gate on it (via `dataTransfer.types`), so a stray file or text drag over
 * the sidebar never lights up a group.
 */
export const CONVERSATION_DRAG_TYPE = "application/x-varagity-conversation";

/** One group with the conversations filed under it, list order preserved. */
export interface GroupSection {
  group: GroupOut;
  conversations: ConversationSummary[];
}

/** The sidebar's partition: group sections first, then the loose list. */
export interface GroupedConversations {
  sections: GroupSection[];
  ungrouped: ConversationSummary[];
}

/**
 * Partition the conversation list by group.
 *
 * Sections follow the group list's (name) order and keep the conversation
 * list's (recency) order within each group; empty groups still get a
 * section — an empty folder is a drop target. A conversation pointing at
 * an unknown group (a stale cache mid-delete) degrades to ungrouped rather
 * than vanishing.
 */
export function groupConversations(
  groups: readonly GroupOut[],
  conversations: readonly ConversationSummary[],
): GroupedConversations {
  const byGroup = new Map<string, ConversationSummary[]>(
    groups.map((group) => [group.group_id, []]),
  );
  const ungrouped: ConversationSummary[] = [];
  for (const conversation of conversations) {
    const filed =
      conversation.group_id != null ? byGroup.get(conversation.group_id) : undefined;
    if (filed) filed.push(conversation);
    else ungrouped.push(conversation);
  }
  return {
    sections: groups.map((group) => ({
      group,
      conversations: byGroup.get(group.group_id) ?? [],
    })),
    ungrouped,
  };
}

/** The conversation id a `/c/{id}` pathname points at, else `null`. */
export function conversationIdFromPathname(pathname: string): string | null {
  const match = /^\/c\/([^/]+)$/.exec(pathname);
  return match ? decodeURIComponent(match[1]) : null;
}

/** The group holding the active conversation, else `null`. */
export function activeGroupId(
  sections: readonly GroupSection[],
  activeConversationId: string | null,
): string | null {
  if (activeConversationId === null) return null;
  for (const section of sections) {
    const holds = section.conversations.some(
      (conversation) => conversation.conversation_id === activeConversationId,
    );
    if (holds) return section.group.group_id;
  }
  return null;
}

/**
 * Whether a group renders expanded: an explicit user toggle always wins;
 * otherwise groups sit collapsed unless they hold the active conversation.
 */
export function isGroupExpanded(
  groupId: string,
  overrides: Readonly<Record<string, boolean>>,
  activeGroup: string | null,
): boolean {
  return overrides[groupId] ?? groupId === activeGroup;
}
