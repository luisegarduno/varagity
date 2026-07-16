"use client";

import { useQueryClient } from "@tanstack/react-query";

import { useMountEffect } from "@/hooks/use-mount-effect";
import { onConversationsChanged } from "@/lib/conversations-bus";
import { queryKeys } from "@/lib/queries";
import { onSettingsChanged } from "@/lib/settings-bus";

// A turn's auto-title is generated in the background and lands a few
// seconds after the turn itself persists, with nothing to announce it —
// so a conversations change gets a second look shortly afterwards.
const AUTO_TITLE_DELAY_MS = 4000;

/**
 * Translates the window buses (`lib/*-bus.ts`) into cache invalidations.
 *
 * Mutating surfaces stay decoupled — they still just call `notify*()` — but
 * the refetching is one concern in one always-mounted place instead of a
 * subscription in every reader. Mounted by the root layout, so an
 * invalidation reaches the cache whether or not the sidebar, the drawer, or
 * the corpus page happens to be on screen.
 */
export function QueryBusBridge() {
  const queryClient = useQueryClient();

  // `queryClient` is a context singleton, so the subscriptions are set up
  // once and torn down once — nothing here goes stale between renders.
  useMountEffect(() => {
    let autoTitleTimer: ReturnType<typeof setTimeout> | undefined;

    const invalidateConversations = () =>
      void queryClient.invalidateQueries({ queryKey: queryKeys.conversations });

    const offConversations = onConversationsChanged(() => {
      invalidateConversations();
      clearTimeout(autoTitleTimer);
      autoTitleTimer = setTimeout(invalidateConversations, AUTO_TITLE_DELAY_MS);
    });

    const offSettings = onSettingsChanged(() => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.settings });
    });

    return () => {
      offConversations();
      offSettings();
      clearTimeout(autoTitleTimer);
    };
  });

  return null;
}
