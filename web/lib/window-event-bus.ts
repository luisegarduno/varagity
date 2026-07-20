/**
 * The shared shape behind every tiny window-event bus (conversations,
 * settings, and the UI keyboard/palette seam): a named `window` event with a
 * `notify` emitter and an `on` subscriber. Each bus file builds its
 * `notify*`/`on*` pair from one call, so the "no-op server-side, subscribe
 * returns its unsubscribe" contract lives in exactly one place.
 */

/** A `notify` emitter and an `on` subscriber over one named window event. */
export interface WindowEventBus {
  /** Dispatch the event. No-op server-side (no `window`). */
  notify: () => void;
  /** Subscribe to the event; returns the unsubscribe. */
  on: (listener: () => void) => () => void;
}

/**
 * Build a bus over `eventName`. `notify` guards on `window` so it is safe to
 * call during SSR; `on` attaches a plain listener and hands back its
 * removal — the same semantics every bus wrote by hand before.
 */
export function createWindowEventBus(eventName: string): WindowEventBus {
  return {
    notify() {
      if (typeof window !== "undefined") {
        window.dispatchEvent(new Event(eventName));
      }
    },
    on(listener) {
      window.addEventListener(eventName, listener);
      return () => window.removeEventListener(eventName, listener);
    },
  };
}
