import { afterEach, describe, expect, it, vi } from "vitest";

import {
  developerMode,
  developerModeServer,
  evidenceRailOpen,
  evidenceRailOpenServer,
  setDeveloperMode,
  setEvidenceRailOpen,
} from "@/lib/display-prefs";

const CHANGE_EVENT = "varagity:display-prefs-changed";
const DEVELOPER_MODE_KEY = "varagity:developer-mode";
const EVIDENCE_RAIL_OPEN_KEY = "varagity:evidence-rail-open";

interface FakeWindow {
  store: Map<string, string>;
  events: string[];
}

/**
 * Stub the module's `window` global with an in-memory localStorage and a
 * dispatch spy. `throwOn` forces the matching localStorage method to throw,
 * standing in for privacy mode. Undone by `vi.unstubAllGlobals()` in
 * afterEach, which returns `window` to undefined — the SSR case.
 */
function installWindow(
  options: { seed?: Record<string, string>; throwOn?: "get" | "set" } = {},
): FakeWindow {
  const store = new Map<string, string>(Object.entries(options.seed ?? {}));
  const events: string[] = [];
  vi.stubGlobal("window", {
    localStorage: {
      getItem(key: string): string | null {
        if (options.throwOn === "get") throw new Error("blocked");
        return store.has(key) ? (store.get(key) as string) : null;
      },
      setItem(key: string, value: string): void {
        if (options.throwOn === "set") throw new Error("blocked");
        store.set(key, value);
      },
    },
    dispatchEvent(event: Event): boolean {
      events.push(event.type);
      return true;
    },
  });
  return { store, events };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

// developerMode and evidenceRailOpen share the "default-on" idiom
// (`getItem(...) !== "false"`, `catch → true`, server snapshot true), so the
// same matrix pins both — the second happens to be the file's previously
// untested existing pref.
const defaultOnPrefs = [
  {
    name: "developerMode",
    key: DEVELOPER_MODE_KEY,
    read: developerMode,
    server: developerModeServer,
    write: setDeveloperMode,
  },
  {
    name: "evidenceRailOpen",
    key: EVIDENCE_RAIL_OPEN_KEY,
    read: evidenceRailOpen,
    server: evidenceRailOpenServer,
    write: setEvidenceRailOpen,
  },
] as const;

describe.each(defaultOnPrefs)(
  "$name (default-on display pref)",
  ({ key, read, server, write }) => {
    it("defaults on when the key is absent", () => {
      installWindow();
      expect(read()).toBe(true);
    });

    it('reads off only for the literal "false"', () => {
      installWindow({ seed: { [key]: "false" } });
      expect(read()).toBe(false);
    });

    it('reads on for "true" and for junk values', () => {
      installWindow({ seed: { [key]: "true" } });
      expect(read()).toBe(true);
      installWindow({ seed: { [key]: "yes-please" } });
      expect(read()).toBe(true);
    });

    it("round-trips through the setter and notifies readers", () => {
      const { store, events } = installWindow();
      write(false);
      expect(store.get(key)).toBe("false");
      expect(read()).toBe(false);
      write(true);
      expect(store.get(key)).toBe("true");
      expect(read()).toBe(true);
      expect(events).toEqual([CHANGE_EVENT, CHANGE_EVENT]);
    });

    it("falls back to the default when localStorage.getItem throws", () => {
      installWindow({ seed: { [key]: "false" }, throwOn: "get" });
      expect(read()).toBe(true);
    });

    it("falls back to the default when window is undefined (SSR)", () => {
      // No installWindow(): `window` is not a global in the node env, so the
      // reference throws inside the try and the catch returns the default.
      expect(read()).toBe(true);
    });

    it("dispatches the change event even when the write throws", () => {
      const { store, events } = installWindow({ throwOn: "set" });
      write(false);
      expect(store.has(key)).toBe(false); // the persist was swallowed…
      expect(events).toEqual([CHANGE_EVENT]); // …but readers still hear it
    });

    it("has an on server-render snapshot", () => {
      expect(server()).toBe(true);
    });
  },
);
