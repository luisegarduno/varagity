/**
 * The TanStack Query layer over {@link file://./api.ts}.
 *
 * One `queryOptions` factory per server dataset, so every consumer of the
 * same data shares one cache entry and one in-flight request: the sidebar
 * and the ⌘K palette read the same conversation list, the settings drawer
 * and the composer's quick-toggles the same catalog. Mutations elsewhere
 * announce themselves on the window buses (`lib/*-bus.ts`), whose
 * subscribers turn the event into `invalidateQueries` against these keys —
 * the buses stayed the seam, the refetching moved into the cache.
 *
 * Keys are namespaced so the conversation *list* and a conversation's
 * *transcript* invalidate independently: a persisted turn re-orders the
 * list without discarding the open transcript that was just folded into
 * the cache.
 */
import { queryOptions } from "@tanstack/react-query";

import {
  getConfig,
  getConversation,
  getSettings,
  listConversations,
  listDocuments,
  listGroups,
  locatePreview,
} from "@/lib/api";

/**
 * Every query key the app uses. Exported (and asserted in the unit suite)
 * because a typo here silently breaks invalidation rather than failing.
 */
export const queryKeys = {
  /** The conversation list — deliberately disjoint from `conversation`. */
  conversations: ["conversations", "list"] as const,
  /** One conversation's transcript. */
  conversation: (id: string) => ["conversations", "detail", id] as const,
  /** The sidebar's conversation groups (folders). */
  groups: ["groups"] as const,
  config: ["config"] as const,
  settings: ["settings"] as const,
  documents: ["documents"] as const,
  /** One chunk's located page preview (keyed by chunk, not its text). */
  preview: (docId: string, chunkKey: string) =>
    ["preview", docId, chunkKey] as const,
};

/** The conversation list, most recently updated first (the sidebar, ⌘K). */
export function conversationsQuery() {
  return queryOptions({
    queryKey: queryKeys.conversations,
    queryFn: () => listConversations(),
  });
}

/** One conversation's full transcript, including each answer's sources. */
export function conversationQuery(id: string) {
  return queryOptions({
    queryKey: queryKeys.conversation(id),
    queryFn: () => getConversation(id),
  });
}

/** The sidebar's conversation groups, name order (the folder list). */
export function groupsQuery() {
  return queryOptions({
    queryKey: queryKeys.groups,
    queryFn: () => listGroups(),
  });
}

/**
 * Static capabilities + upload constraints. These are fixed for the life of
 * the API process, so the cache never needs to look again.
 */
export function configQuery() {
  return queryOptions({
    queryKey: queryKeys.config,
    queryFn: () => getConfig(),
    staleTime: Infinity,
  });
}

/** The effective settings catalog + the corpus-stale flag (spec_v2 §4.7). */
export function settingsQuery() {
  return queryOptions({
    queryKey: queryKeys.settings,
    queryFn: () => getSettings(),
  });
}

/** The ingested-document table. */
export function documentsQuery() {
  return queryOptions({
    queryKey: queryKeys.documents,
    queryFn: () => listDocuments(),
  });
}

/**
 * Where one chunk's text lives in its source document (ADR-010). The
 * preview component mounts on first expand — mounting is the fetch
 * trigger — and `doc_id` is content-hashed, so a located page can never
 * go stale: cache it for the life of the tab and re-expand instantly.
 */
export function previewQuery(docId: string, chunkKey: string, text: string) {
  return queryOptions({
    queryKey: queryKeys.preview(docId, chunkKey),
    queryFn: () => locatePreview(docId, text),
    staleTime: Infinity,
    retry: 1,
  });
}
