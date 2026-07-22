/**
 * A tiny window-event bus announcing conversation-group changes (create,
 * delete); only `QueryBusBridge` subscribes, turning each event into
 * `invalidateQueries`.
 */
import { createWindowEventBus } from "@/lib/window-event-bus";

const bus = createWindowEventBus("varagity:groups-changed");

/** Announce that the group list changed. No-op server-side. */
export function notifyGroupsChanged(): void {
  bus.notify();
}

/** Subscribe to group-list changes; returns the unsubscribe. */
export function onGroupsChanged(listener: () => void): () => void {
  return bus.on(listener);
}
