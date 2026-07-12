/**
 * A tiny window-event bus so the sidebar can refetch its conversation list
 * when any component mutates conversations (new chat, turn persisted,
 * delete) without prop-drilling or a data library — enough for the thin
 * slice; Phase 8+ can move to a fetch cache if the surfaces multiply.
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
