/**
 * The keyboard/palette seam: a tiny window-event bus connecting
 * global input surfaces (keyboard shortcuts, later the command palette) to
 * the components that own the corresponding UI state — the settings drawer,
 * the evidence rail, the composer. Emitters fire `notify*()`; the owning
 * component subscribes with `on*()` and performs its own state change.
 * Same shape as settings-bus, enough for the single-user surface.
 */
const OPEN_SETTINGS_EVENT = "varagity:open-settings";
const TOGGLE_EVIDENCE_EVENT = "varagity:toggle-evidence";
const FOCUS_COMPOSER_EVENT = "varagity:focus-composer";

/** Ask the settings drawer to open. No-op server-side. */
export function notifyOpenSettings(): void {
  if (typeof window !== "undefined") {
    window.dispatchEvent(new Event(OPEN_SETTINGS_EVENT));
  }
}

/** Subscribe to open-settings requests; returns the unsubscribe. */
export function onOpenSettings(listener: () => void): () => void {
  window.addEventListener(OPEN_SETTINGS_EVENT, listener);
  return () => window.removeEventListener(OPEN_SETTINGS_EVENT, listener);
}

/** Ask the evidence rail to toggle. No-op server-side. */
export function notifyToggleEvidence(): void {
  if (typeof window !== "undefined") {
    window.dispatchEvent(new Event(TOGGLE_EVIDENCE_EVENT));
  }
}

/** Subscribe to evidence-toggle requests; returns the unsubscribe. */
export function onToggleEvidence(listener: () => void): () => void {
  window.addEventListener(TOGGLE_EVIDENCE_EVENT, listener);
  return () => window.removeEventListener(TOGGLE_EVIDENCE_EVENT, listener);
}

/** Ask the chat composer to take focus. No-op server-side. */
export function notifyFocusComposer(): void {
  if (typeof window !== "undefined") {
    window.dispatchEvent(new Event(FOCUS_COMPOSER_EVENT));
  }
}

/** Subscribe to focus-composer requests; returns the unsubscribe. */
export function onFocusComposer(listener: () => void): () => void {
  window.addEventListener(FOCUS_COMPOSER_EVENT, listener);
  return () => window.removeEventListener(FOCUS_COMPOSER_EVENT, listener);
}
