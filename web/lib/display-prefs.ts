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
const CHANGE_EVENT = "varagity:display-prefs-changed";

/** The curated accent hues (spec_v2 §4.7 Display "accent"); CSS owns the colors. */
export const ACCENTS = ["indigo", "teal", "violet", "ember"] as const;
export type Accent = (typeof ACCENTS)[number];
export const DEFAULT_ACCENT: Accent = "indigo";

/** Layout densities (spec_v2 §4.7 Display "density"). */
export const DENSITIES = ["comfortable", "compact"] as const;
export type Density = (typeof DENSITIES)[number];
export const DEFAULT_DENSITY: Density = "comfortable";

/** Whether finished answers' reasoning traces start expanded. */
export function reasoningDefaultOpen(): boolean {
  try {
    return window.localStorage.getItem(REASONING_DEFAULT_OPEN_KEY) === "true";
  } catch {
    return false; // storage unavailable (SSR, privacy mode) — the default
  }
}

/** The server-render snapshot (no localStorage there). */
export function reasoningDefaultOpenServer(): boolean {
  return false;
}

/** Persist the reasoning-trace default (best-effort) and notify readers. */
export function setReasoningDefaultOpen(open: boolean): void {
  try {
    window.localStorage.setItem(REASONING_DEFAULT_OPEN_KEY, String(open));
  } catch {
    // Storage unavailable — the toggle simply doesn't persist.
  }
  window.dispatchEvent(new Event(CHANGE_EVENT));
}

/** The chosen accent hue (falls back to the default on junk/absence). */
export function accent(): Accent {
  try {
    const raw = window.localStorage.getItem(ACCENT_KEY);
    return (ACCENTS as readonly string[]).includes(raw ?? "")
      ? (raw as Accent)
      : DEFAULT_ACCENT;
  } catch {
    return DEFAULT_ACCENT;
  }
}

/** The server-render snapshot for the accent. */
export function accentServer(): Accent {
  return DEFAULT_ACCENT;
}

/** Persist the accent (best-effort) and notify readers. */
export function setAccent(value: Accent): void {
  try {
    window.localStorage.setItem(ACCENT_KEY, value);
  } catch {
    // Storage unavailable — the choice simply doesn't persist.
  }
  window.dispatchEvent(new Event(CHANGE_EVENT));
}

/** The chosen layout density (falls back to the default on junk/absence). */
export function density(): Density {
  try {
    const raw = window.localStorage.getItem(DENSITY_KEY);
    return (DENSITIES as readonly string[]).includes(raw ?? "")
      ? (raw as Density)
      : DEFAULT_DENSITY;
  } catch {
    return DEFAULT_DENSITY;
  }
}

/** The server-render snapshot for the density. */
export function densityServer(): Density {
  return DEFAULT_DENSITY;
}

/** Persist the density (best-effort) and notify readers. */
export function setDensity(value: Density): void {
  try {
    window.localStorage.setItem(DENSITY_KEY, value);
  } catch {
    // Storage unavailable — the choice simply doesn't persist.
  }
  window.dispatchEvent(new Event(CHANGE_EVENT));
}

/** Whether the desktop evidence rail is expanded (collapse survives reloads). */
export function evidenceRailOpen(): boolean {
  try {
    return window.localStorage.getItem(EVIDENCE_RAIL_OPEN_KEY) !== "false";
  } catch {
    return true; // storage unavailable (SSR, privacy mode) — the default
  }
}

/** The server-render snapshot for the evidence rail (open). */
export function evidenceRailOpenServer(): boolean {
  return true;
}

/** Persist the evidence-rail state (best-effort) and notify readers. */
export function setEvidenceRailOpen(open: boolean): void {
  try {
    window.localStorage.setItem(EVIDENCE_RAIL_OPEN_KEY, String(open));
  } catch {
    // Storage unavailable — the toggle simply doesn't persist.
  }
  window.dispatchEvent(new Event(CHANGE_EVENT));
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
