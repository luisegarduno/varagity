import { QueryClient } from "@tanstack/react-query";
import { describe, expect, it } from "vitest";

import {
  configQuery,
  conversationQuery,
  conversationsQuery,
  documentsQuery,
  graphDocumentsQuery,
  graphStatusQuery,
  groupsQuery,
  previewQuery,
  queryKeys,
  settingsQuery,
} from "@/lib/queries";

describe("query keys", () => {
  it("points each dataset at its key", () => {
    expect(conversationsQuery().queryKey).toEqual(queryKeys.conversations);
    expect(conversationQuery("abc").queryKey).toEqual(
      queryKeys.conversation("abc"),
    );
    expect(groupsQuery().queryKey).toEqual(queryKeys.groups);
    expect(configQuery().queryKey).toEqual(queryKeys.config);
    expect(settingsQuery().queryKey).toEqual(queryKeys.settings);
    expect(documentsQuery().queryKey).toEqual(queryKeys.documents);
    expect(graphStatusQuery().queryKey).toEqual(queryKeys.graphStatus);
    expect(graphDocumentsQuery().queryKey).toEqual(queryKeys.graphDocuments);
    expect(previewQuery("d0", "d0::3", "text").queryKey).toEqual(
      queryKeys.preview("d0", "d0::3"),
    );
  });

  it("keeps the group list disjoint from the conversation keys", () => {
    // Group CRUD invalidates `groups`; a move invalidates the conversation
    // list. Neither may prefix-match the other's cache entries.
    const client = new QueryClient();
    client.setQueryData(queryKeys.groups, []);
    client.setQueryData(queryKeys.conversations, []);

    const matched = client
      .getQueryCache()
      .findAll({ queryKey: queryKeys.groups })
      .map((query) => query.queryKey);

    expect(matched).toEqual([queryKeys.groups]);
  });

  it("keeps one conversation's key distinct from another's", () => {
    expect(queryKeys.conversation("a")).not.toEqual(queryKeys.conversation("b"));
  });

  it("keys previews per chunk, not per locate text", () => {
    // The chunk id embeds the content-hashed doc id, so it fully names the
    // result; hanging the (up to 20 kB) chunk text off the key would only
    // bloat every cache lookup.
    expect(queryKeys.preview("d0", "d0::3")).toEqual(["preview", "d0", "d0::3"]);
    expect(queryKeys.preview("d0", "d0::3")).not.toEqual(
      queryKeys.preview("d0", "d0::4"),
    );
  });

  it("keeps the conversation list disjoint from a transcript", () => {
    // Invalidation is prefix-matched. If the list's key were a prefix of a
    // transcript's, then every persisted turn — which invalidates the list
    // to re-order it — would also discard the transcript that turn was just
    // folded into, and re-fetch it for nothing.
    const client = new QueryClient();
    client.setQueryData(queryKeys.conversations, []);
    client.setQueryData(queryKeys.conversation("abc"), { messages: [] });

    const matched = client
      .getQueryCache()
      .findAll({ queryKey: queryKeys.conversations })
      .map((query) => query.queryKey);

    expect(matched).toEqual([queryKeys.conversations]);
  });
});

describe("configQuery", () => {
  it("never re-asks, since capabilities are fixed for the API's lifetime", () => {
    expect(configQuery().staleTime).toBe(Infinity);
  });
});

describe("graph queries", () => {
  it("keeps the graph's size and its corpus listing disjoint", () => {
    // A build changes the graph but not the corpus directory; an upload
    // the other way round. Invalidation is prefix-matched, so neither key
    // may be a prefix of the other's.
    const client = new QueryClient();
    client.setQueryData(queryKeys.graphStatus, {});
    client.setQueryData(queryKeys.graphDocuments, []);

    const matched = client
      .getQueryCache()
      .findAll({ queryKey: queryKeys.graphStatus })
      .map((query) => query.queryKey);

    expect(matched).toEqual([queryKeys.graphStatus]);
  });

  it("goes stale quickly — a running build grows the counts as you watch", () => {
    expect(graphStatusQuery().staleTime).toBe(5_000);
  });
});

describe("previewQuery", () => {
  it("never goes stale — doc_id is content-hashed, so a located page is final", () => {
    const query = previewQuery("a398491c7441925f", "a398491c7441925f::0", "x");
    expect(query.staleTime).toBe(Infinity);
    expect(query.retry).toBe(1);
  });
});
