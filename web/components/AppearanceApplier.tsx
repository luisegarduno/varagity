"use client";

import { useEffect, useSyncExternalStore } from "react";

import {
  DEFAULT_ACCENT,
  DEFAULT_DENSITY,
  accent,
  accentServer,
  density,
  densityServer,
  subscribeDisplayPrefs,
} from "@/lib/display-prefs";

/**
 * Stamps the display preferences onto <html> as `data-accent` /
 * `data-density` attributes — the selectors globals.css themes against.
 * Defaults are expressed by absence so the resting DOM stays clean.
 * Renders nothing; theme itself is owned by next-themes.
 */
export function AppearanceApplier() {
  const currentAccent = useSyncExternalStore(
    subscribeDisplayPrefs,
    accent,
    accentServer,
  );
  const currentDensity = useSyncExternalStore(
    subscribeDisplayPrefs,
    density,
    densityServer,
  );

  useEffect(() => {
    const root = document.documentElement;
    if (currentAccent === DEFAULT_ACCENT) {
      root.removeAttribute("data-accent");
    } else {
      root.setAttribute("data-accent", currentAccent);
    }
  }, [currentAccent]);

  useEffect(() => {
    const root = document.documentElement;
    if (currentDensity === DEFAULT_DENSITY) {
      root.removeAttribute("data-density");
    } else {
      root.setAttribute("data-density", currentDensity);
    }
  }, [currentDensity]);

  return null;
}
