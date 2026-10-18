/**
 * Where the map's vendored favicons are served from.
 *
 * The icons were fetched once at authoring time into `public/map-icons/`
 * (the map never fetches icons from the network — house privacy rule), and
 * the Next app serves that folder at the site root, so the default prefix is
 * absolute. The docs-site bundle (`web/docs-map/`, ADR-018) ships the same
 * PNGs beside its own page and provides a *relative* prefix instead: GitHub
 * Pages serves the site under a repository subpath, where `/map-icons/…`
 * would miss. Every icon `src` resolves through {@link useIconSrc}; nothing
 * else in the map knows the path.
 */

import { createContext, useContext } from "react";

/** URL prefix, without a trailing slash, under which `<domain>.png` lives. */
export const MapIconBaseContext = createContext("/map-icons");

/** Resolve a favicon domain to its vendored asset under `base`. */
export function iconSrc(base: string, domain: string): string {
  return `${base}/${domain}.png`;
}

/** The icon resolver bound to the enclosing {@link MapIconBaseContext}. */
export function useIconSrc(): (domain: string) => string {
  const base = useContext(MapIconBaseContext);
  return (domain) => iconSrc(base, domain);
}
