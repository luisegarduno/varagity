/**
 * A tiny window-event bus announcing conversation-list changes (new chat,
 * turn persisted, delete); only `QueryBusBridge` subscribes, turning each
 * event into `invalidateQueries`.
 */
const EVENT_NAME = "varagity:conversations-changed";

/** Announce that the conversation list changed. No-op server-side. */
export function notifyConversationsChanged(): void {
  if (typeof window !== "undefined") {
    window.dispatchEvent(new Event(EVENT_NAME));
  }
}

/** Subscribe to conversation-list changes; returns the unsubscribe. */
export function onConversationsChanged(listener: () => void): () => void {
  window.addEventListener(EVENT_NAME, listener);
  return () => window.removeEventListener(EVENT_NAME, listener);
}
