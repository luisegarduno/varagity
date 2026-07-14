/**
 * A tiny window-event bus so every settings consumer (drawer, composer
 * quick-toggles, the sidebar's stale indicator) refetches when any of them
 * PATCHes the shared override layer — the same pattern as the
 * conversations bus, enough for the single-user surface.
 */
const EVENT_NAME = "varagity:settings-changed";

/** Announce that the runtime settings changed. No-op server-side. */
export function notifySettingsChanged(): void {
  if (typeof window !== "undefined") {
    window.dispatchEvent(new Event(EVENT_NAME));
  }
}

/** Subscribe to settings changes; returns the unsubscribe. */
export function onSettingsChanged(listener: () => void): () => void {
  window.addEventListener(EVENT_NAME, listener);
  return () => window.removeEventListener(EVENT_NAME, listener);
}
