import { afterEach, describe, expect, it, vi } from "vitest";

import { createWindowEventBus } from "@/lib/window-event-bus";

// A real EventTarget as `window` so dispatch actually reaches listeners;
// unstubbed back to undefined (the SSR case) after each test, mirroring the
// display-prefs suite's stubbing style.
afterEach(() => {
  vi.unstubAllGlobals();
});

describe("createWindowEventBus", () => {
  it("notify dispatches the named event that on receives", () => {
    vi.stubGlobal("window", new EventTarget());
    const bus = createWindowEventBus("varagity:test-event");
    let received = 0;
    bus.on(() => {
      received += 1;
    });
    bus.notify();
    expect(received).toBe(1);
  });

  it("unsubscribe stops further delivery", () => {
    vi.stubGlobal("window", new EventTarget());
    const bus = createWindowEventBus("varagity:test-event");
    let received = 0;
    const off = bus.on(() => {
      received += 1;
    });
    bus.notify();
    off();
    bus.notify();
    expect(received).toBe(1);
  });

  it("only delivers to listeners of the same event name", () => {
    vi.stubGlobal("window", new EventTarget());
    const a = createWindowEventBus("varagity:a");
    const b = createWindowEventBus("varagity:b");
    let aCount = 0;
    a.on(() => {
      aCount += 1;
    });
    b.notify();
    expect(aCount).toBe(0);
  });

  it("notify without a window does not throw (SSR)", () => {
    // No stub: `window` is undefined in the node env, so notify no-ops.
    expect(() => createWindowEventBus("varagity:test-event").notify()).not.toThrow();
  });
});
