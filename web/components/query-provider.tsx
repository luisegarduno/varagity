"use client";

import {
  environmentManager,
  QueryClient,
  QueryClientProvider,
} from "@tanstack/react-query";

function makeQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        // The API is on localhost and every failure here is actionable and
        // immediate — a dead stack, an unknown conversation, a rejected
        // patch. Retrying would only delay the banner that tells the user
        // to bring the stack up, so failures surface on the first attempt.
        retry: false,
        // Long enough that opening ⌘K reuses the sidebar's list instead of
        // refetching it; short enough that a window-focus refetch (the
        // default) picks up out-of-band changes like a CLI ingest. Neither
        // bounds correctness: the buses invalidate on every mutation, and
        // invalidation refetches regardless of staleness.
        staleTime: 30_000,
      },
    },
  });
}

let browserQueryClient: QueryClient | undefined;

function getQueryClient() {
  // Server: a fresh client per request, so no cache is ever shared between
  // users. Browser: one singleton, and notably *not* useState — React
  // throws the initial-render client away if something suspends beneath a
  // provider with no boundary in between.
  if (environmentManager.isServer()) return makeQueryClient();
  browserQueryClient ??= makeQueryClient();
  return browserQueryClient;
}

/**
 * The TanStack Query cache for the whole app (`lib/queries.ts` holds the
 * datasets). Client-side because `QueryClientProvider` is context; every
 * fetch in this app is browser-originated anyway, since the API origin is
 * the browser's own `NEXT_PUBLIC_API_URL` rather than the web container's.
 */
export function QueryProvider({ children }: { children: React.ReactNode }) {
  return (
    <QueryClientProvider client={getQueryClient()}>
      {children}
    </QueryClientProvider>
  );
}
