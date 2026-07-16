import { QueryClient } from "@tanstack/react-query";
import { describe, expect, it } from "vitest";

import {
  configQuery,
  conversationQuery,
  conversationsQuery,
  documentsQuery,
  queryKeys,
  settingsQuery,
} from "@/lib/queries";

describe("query keys", () => {
  it("points each dataset at its key", () => {
    expect(conversationsQuery().queryKey).toEqual(queryKeys.conversations);
    expect(conversationQuery("abc").queryKey).toEqual(
      queryKeys.conversation("abc"),
    );
    expect(configQuery().queryKey).toEqual(queryKeys.config);
    expect(settingsQuery().queryKey).toEqual(queryKeys.settings);
    expect(documentsQuery().queryKey).toEqual(queryKeys.documents);
  });

  it("keeps one conversation's key distinct from another's", () => {
    expect(queryKeys.conversation("a")).not.toEqual(queryKeys.conversation("b"));
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
