"use client";

import { useEffect } from "react";

/**
 * Run `effect` once on mount; its return value cleans up on unmount.
 *
 * The one sanctioned `useEffect` in the codebase (see the `no-use-effect`
 * skill and the `no-restricted-syntax` rule that enforces it). Reach for it
 * only to synchronize with an external system whose lifecycle really is
 * "set up on mount, tear down on unmount": DOM integration, browser-API
 * subscriptions, third-party widget handles.
 *
 * Everything the effect closes over is captured once, so read only stable
 * references inside — refs, `useCallback(…, [])` callbacks, context
 * singletons like the QueryClient. To re-run against a value that changes,
 * `key` the component on that value and let it remount rather than
 * reaching back for a dependency array.
 */
export function useMountEffect(effect: () => void | (() => void)): void {
  // The rule exists to push callers here; this is the escape hatch itself.
  // exhaustive-deps can't see that the empty array is the whole point.
  // eslint-disable-next-line no-restricted-syntax, react-hooks/exhaustive-deps
  useEffect(effect, []);
}
