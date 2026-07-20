/**
 * A tiny window-event bus so every settings consumer (drawer, composer
 * quick-toggles, the sidebar's stale indicator) refetches when any of them
 * PATCHes the shared override layer — the same pattern as the
 * conversations bus, enough for the single-user surface.
 */
import { createWindowEventBus } from "@/lib/window-event-bus";

const bus = createWindowEventBus("varagity:settings-changed");

/** Announce that the runtime settings changed. No-op server-side. */
export function notifySettingsChanged(): void {
  bus.notify();
}

/** Subscribe to settings changes; returns the unsubscribe. */
export function onSettingsChanged(listener: () => void): () => void {
  return bus.on(listener);
}
