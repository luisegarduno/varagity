/**
 * The keyboard/palette seam: a tiny window-event bus connecting
 * global input surfaces (keyboard shortcuts, later the command palette) to
 * the components that own the corresponding UI state — the settings drawer,
 * the evidence rail, the composer. Emitters fire `notify*()`; the owning
 * component subscribes with `on*()` and performs its own state change.
 * Same shape as settings-bus, enough for the single-user surface.
 */
import { createWindowEventBus } from "@/lib/window-event-bus";

const openSettings = createWindowEventBus("varagity:open-settings");
const toggleEvidence = createWindowEventBus("varagity:toggle-evidence");
const focusComposer = createWindowEventBus("varagity:focus-composer");

/** Ask the settings drawer to open. No-op server-side. */
export function notifyOpenSettings(): void {
  openSettings.notify();
}

/** Subscribe to open-settings requests; returns the unsubscribe. */
export function onOpenSettings(listener: () => void): () => void {
  return openSettings.on(listener);
}

/** Ask the evidence rail to toggle. No-op server-side. */
export function notifyToggleEvidence(): void {
  toggleEvidence.notify();
}

/** Subscribe to evidence-toggle requests; returns the unsubscribe. */
export function onToggleEvidence(listener: () => void): () => void {
  return toggleEvidence.on(listener);
}

/** Ask the chat composer to take focus. No-op server-side. */
export function notifyFocusComposer(): void {
  focusComposer.notify();
}

/** Subscribe to focus-composer requests; returns the unsubscribe. */
export function onFocusComposer(listener: () => void): () => void {
  return focusComposer.on(listener);
}
