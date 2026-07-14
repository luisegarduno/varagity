/**
 * Client-side display preferences (spec_v2 §4.7's Display group). These
 * never reach the pipeline, so they live in localStorage instead of the
 * server-side override layer; theme itself is owned by next-themes.
 *
 * Shaped as an external store (snapshot + subscribe) so components read
 * it with `useSyncExternalStore` — SSR-safe (the server snapshot is the
 * default) and live across components (the drawer's toggle re-renders
 * every open reasoning trace).
 */
const REASONING_DEFAULT_OPEN_KEY = "varagity:reasoning-default-open";
const CHANGE_EVENT = "varagity:display-prefs-changed";

/** Whether finished answers' reasoning traces start expanded. */
export function reasoningDefaultOpen(): boolean {
  try {
    return window.localStorage.getItem(REASONING_DEFAULT_OPEN_KEY) === "true";
  } catch {
    return false; // storage unavailable (SSR, privacy mode) — the default
  }
}

/** The server-render snapshot (no localStorage there). */
export function reasoningDefaultOpenServer(): boolean {
  return false;
}

/** Persist the reasoning-trace default (best-effort) and notify readers. */
export function setReasoningDefaultOpen(open: boolean): void {
  try {
    window.localStorage.setItem(REASONING_DEFAULT_OPEN_KEY, String(open));
  } catch {
    // Storage unavailable — the toggle simply doesn't persist.
  }
  window.dispatchEvent(new Event(CHANGE_EVENT));
}

/** Subscribe to preference changes (this tab + other tabs' storage events). */
export function subscribeDisplayPrefs(listener: () => void): () => void {
  window.addEventListener(CHANGE_EVENT, listener);
  window.addEventListener("storage", listener);
  return () => {
    window.removeEventListener(CHANGE_EVENT, listener);
    window.removeEventListener("storage", listener);
  };
}
