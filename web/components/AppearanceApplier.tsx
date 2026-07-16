"use client";

import { useSyncExternalStore } from "react";

import { useMountEffect } from "@/hooks/use-mount-effect";
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
 * Stamps one `data-*` attribute onto <html>, or clears it when the value is
 * the default. The caller keys this on `value`, so each distinct preference
 * mounts a fresh instance and the write stays a mount-time sync with the
 * document rather than an effect chasing a dependency.
 */
function RootAttribute({
  name,
  value,
  isDefault,
}: {
  name: string;
  value: string;
  isDefault: boolean;
}) {
  useMountEffect(() => {
    const root = document.documentElement;
    if (isDefault) {
      root.removeAttribute(name);
    } else {
      root.setAttribute(name, value);
    }
  });
  return null;
}

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

  return (
    <>
      <RootAttribute
        key={`accent:${currentAccent}`}
        name="data-accent"
        value={currentAccent}
        isDefault={currentAccent === DEFAULT_ACCENT}
      />
      <RootAttribute
        key={`density:${currentDensity}`}
        name="data-density"
        value={currentDensity}
        isDefault={currentDensity === DEFAULT_DENSITY}
      />
    </>
  );
}
