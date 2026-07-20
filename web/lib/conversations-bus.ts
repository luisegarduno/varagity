/**
 * A tiny window-event bus announcing conversation-list changes (new chat,
 * turn persisted, delete); only `QueryBusBridge` subscribes, turning each
 * event into `invalidateQueries`.
 */
import { createWindowEventBus } from "@/lib/window-event-bus";

const bus = createWindowEventBus("varagity:conversations-changed");

/** Announce that the conversation list changed. No-op server-side. */
export function notifyConversationsChanged(): void {
  bus.notify();
}

/** Subscribe to conversation-list changes; returns the unsubscribe. */
export function onConversationsChanged(listener: () => void): () => void {
  return bus.on(listener);
}
