/**
 * Client-side display preferences (spec_v2 §4.7's Display group). These
 * never reach the pipeline, so they live in localStorage instead of the
 * server-side override layer; theme itself is owned by next-themes.
 *
 * Shaped as an external store (snapshot + subscribe) so components read
 * it with `useSyncExternalStore` — SSR-safe (the server snapshot is the
 * default) and live across components (the drawer's toggle re-renders
 * every open reasoning trace).
 */
const REASONING_DEFAULT_OPEN_KEY = "varagity:reasoning-default-open";
const ACCENT_KEY = "varagity:accent";
const DENSITY_KEY = "varagity:density";
const EVIDENCE_RAIL_OPEN_KEY = "varagity:evidence-rail-open";
const DEVELOPER_MODE_KEY = "varagity:developer-mode";
const CHANGE_EVENT = "varagity:display-prefs-changed";

/** The curated accent hues (spec_v2 §4.7 Display "accent"); CSS owns the colors. */
export const ACCENTS = ["indigo", "teal", "violet", "ember"] as const;
export type Accent = (typeof ACCENTS)[number];
export const DEFAULT_ACCENT: Accent = "indigo";

/** Layout densities (spec_v2 §4.7 Display "density"). */
export const DENSITIES = ["comfortable", "compact"] as const;
export type Density = (typeof DENSITIES)[number];
export const DEFAULT_DENSITY: Density = "comfortable";

/** One persisted preference's read / server-snapshot / setter triple. */
interface Pref<T> {
  get: () => T;
  getServer: () => T;
  set: (value: T) => void;
}

/**
 * Build one preference's accessor triple over a localStorage key. `parse`
 * maps the raw stored string (`null` when absent) to a value; `serialize`
 * is its inverse for the setter. Every getter is storage-safe (SSR, privacy
 * mode → the fallback) and every setter dispatches `CHANGE_EVENT`, so the
 * five prefs below share one implementation of that contract.
 */
function makePref<T>(
  key: string,
  fallback: T,
  parse: (raw: string | null) => T,
  serialize: (value: T) => string,
): Pref<T> {
  return {
    get() {
      try {
        return parse(window.localStorage.getItem(key));
      } catch {
        return fallback; // storage unavailable (SSR, privacy mode) — the default
      }
    },
    getServer() {
      return fallback;
    },
    set(value) {
      try {
        window.localStorage.setItem(key, serialize(value));
      } catch {
        // Storage unavailable — the choice simply doesn't persist.
      }
      window.dispatchEvent(new Event(CHANGE_EVENT));
    },
  };
}

/**
 * A boolean pref stored as `"true"`/`"false"`. `defaultOn` sets both the
 * fallback and the read polarity: default-on reads off only for the literal
 * `"false"`; default-off reads on only for the literal `"true"`.
 */
function booleanPref(key: string, defaultOn: boolean): Pref<boolean> {
  return makePref<boolean>(
    key,
    defaultOn,
    defaultOn ? (raw) => raw !== "false" : (raw) => raw === "true",
    String,
  );
}

/** An enum pref validated against its allowed values (junk/absence → default). */
function enumPref<T extends string>(
  key: string,
  allowed: readonly T[],
  fallback: T,
): Pref<T> {
  return makePref<T>(
    key,
    fallback,
    (raw) => ((allowed as readonly string[]).includes(raw ?? "") ? (raw as T) : fallback),
    (value) => value,
  );
}

const reasoning = booleanPref(REASONING_DEFAULT_OPEN_KEY, false);
const accentPref = enumPref(ACCENT_KEY, ACCENTS, DEFAULT_ACCENT);
const densityPref = enumPref(DENSITY_KEY, DENSITIES, DEFAULT_DENSITY);
const evidenceRail = booleanPref(EVIDENCE_RAIL_OPEN_KEY, true);
const developer = booleanPref(DEVELOPER_MODE_KEY, true);

/** Whether finished answers' reasoning traces start expanded. */
export function reasoningDefaultOpen(): boolean {
  return reasoning.get();
}

/** The server-render snapshot (no localStorage there). */
export function reasoningDefaultOpenServer(): boolean {
  return reasoning.getServer();
}

/** Persist the reasoning-trace default (best-effort) and notify readers. */
export function setReasoningDefaultOpen(open: boolean): void {
  reasoning.set(open);
}

/** The chosen accent hue (falls back to the default on junk/absence). */
export function accent(): Accent {
  return accentPref.get();
}

/** The server-render snapshot for the accent. */
export function accentServer(): Accent {
  return accentPref.getServer();
}

/** Persist the accent (best-effort) and notify readers. */
export function setAccent(value: Accent): void {
  accentPref.set(value);
}

/** The chosen layout density (falls back to the default on junk/absence). */
export function density(): Density {
  return densityPref.get();
}

/** The server-render snapshot for the density. */
export function densityServer(): Density {
  return densityPref.getServer();
}

/** Persist the density (best-effort) and notify readers. */
export function setDensity(value: Density): void {
  densityPref.set(value);
}

/** Whether the desktop evidence rail is expanded (collapse survives reloads). */
export function evidenceRailOpen(): boolean {
  return evidenceRail.get();
}

/** The server-render snapshot for the evidence rail (open). */
export function evidenceRailOpenServer(): boolean {
  return evidenceRail.getServer();
}

/** Persist the evidence-rail state (best-effort) and notify readers. */
export function setEvidenceRailOpen(open: boolean): void {
  evidenceRail.set(open);
}

/**
 * Whether developer mode is on (default on). A *cosmetic* gate (ADR-015 /
 * D7): it only shows or hides the sidebar Map button and the ⌘K "Codebase
 * Map" command — `/map` stays reachable by URL even when it is off. That is
 * fine for a single-user local app; there is no route guard and no server
 * pref behind it.
 */
export function developerMode(): boolean {
  return developer.get();
}

/** The server-render snapshot for developer mode (on). */
export function developerModeServer(): boolean {
  return developer.getServer();
}

/** Persist the developer-mode toggle (best-effort) and notify readers. */
export function setDeveloperMode(on: boolean): void {
  developer.set(on);
}

/** Subscribe to preference changes (this tab + other tabs' storage events). */
export function subscribeDisplayPrefs(listener: () => void): () => void {
  window.addEventListener(CHANGE_EVENT, listener);
  window.addEventListener("storage", listener);
  return () => {
    window.removeEventListener(CHANGE_EVENT, listener);
    window.removeEventListener("storage", listener);
  };
}
