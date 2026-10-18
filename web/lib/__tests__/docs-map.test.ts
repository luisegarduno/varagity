/**
 * Drift guard for the docs-site map bundle (ADR-018).
 *
 * `golden-docs/assets/map/` is a checked-in `vite build` of `web/docs-map/`
 * — the `openapi.json` idiom — and this suite fails when the sources it was
 * built from have changed since. Output bytes are never compared (minifiers
 * drift across versions); the manifest's hash of the inputs is. The rest
 * pins the properties the docs site relies on: relative URLs (GitHub Pages
 * serves a subpath), the icons beside the page, no ELK in the browser, and
 * nothing in the bundle for Material's privacy plugin to fetch except the
 * fonts it is meant to localize.
 */

import { existsSync, readdirSync, readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

import {
  DOCS_MAP_OUTPUTS,
  hashInputs,
  listInputFiles,
  REGENERATE_COMMAND,
  type DocsMapManifest,
} from "@/docs-map/manifest";

const WEB_ROOT = path.resolve(import.meta.dirname, "../..");
const BUNDLE_DIR = path.resolve(WEB_ROOT, "../golden-docs/assets/map");

// The exact patterns material/plugins/privacy/plugin.py scans static .css
// and .js files with: a hit is downloaded at docs-build time and rewritten.
const PRIVACY_CSS_URL = /url\(\s*(["']?)(?:https?:)?\/\/[^)'"]+\1\s*\)/gi;
const PRIVACY_JS_URL = /["'](?:https?:)?\/\/[^"']+\.(?:css|js(?:on)?)["']/gi;

const read = (name: string): string =>
  readFileSync(path.join(BUNDLE_DIR, name), "utf8");

describe("docs-site map bundle (golden-docs/assets/map)", () => {
  const manifest = JSON.parse(read("manifest.json")) as DocsMapManifest;

  it("was built from the sources as they are now", () => {
    expect(
      manifest.sourceHash,
      `golden-docs/assets/map is stale — regenerate it with \`${REGENERATE_COMMAND}\``,
    ).toBe(hashInputs(WEB_ROOT));
  });

  it("names the regeneration command the guard quotes", () => {
    expect(manifest.regenerate).toBe(REGENERATE_COMMAND);
  });

  it("hashed the same files the guard hashes", () => {
    expect(manifest.inputs).toEqual(listInputFiles(WEB_ROOT));
  });

  it("hashes enough that the guard can't pass vacuously", () => {
    const files = listInputFiles(WEB_ROOT);
    expect(files.length).toBeGreaterThanOrEqual(20);
    for (const must of [
      "app/globals.css",
      "components/map/CodebaseMapView.tsx",
      "docs-map/index.html",
      "lib/codebase-map.data.ts",
      "lib/codebase-map.layout.json",
      "public/map-icons/postgresql.org.png",
    ]) {
      expect(files).toContain(must);
    }
  });

  it("ships every output and every vendored icon", () => {
    for (const file of DOCS_MAP_OUTPUTS) {
      expect(existsSync(path.join(BUNDLE_DIR, file)), file).toBe(true);
    }
    const icons = readdirSync(path.join(WEB_ROOT, "public/map-icons"));
    expect(icons.length).toBeGreaterThan(0);
    for (const icon of icons) {
      expect(existsSync(path.join(BUNDLE_DIR, "map-icons", icon)), icon).toBe(true);
    }
  });

  it("references everything relatively, icons included", () => {
    const html = read("index.html");
    expect(html).toContain('src="./map.js"');
    expect(html).toContain('href="./map.css"');
    expect(html).toContain('href="fonts.css"');
    expect(html).not.toMatch(/(?:src|href)="\//);
    // The icon base the entry provides (icon-base.ts) survives minification
    // (whatever quote style the minifier picks).
    expect(read("map.js")).toMatch(/[`"']map-icons[`"']/);
  });

  it("ships one script and no ELK (the layout is the checked-in snapshot)", () => {
    const scripts = readdirSync(BUNDLE_DIR).filter((f) => f.endsWith(".js"));
    expect(scripts).toEqual(["map.js"]);
    expect(read("map.js")).not.toMatch(/elk-worker|elk-api/);
  });

  it("gives the privacy plugin nothing to fetch but the fonts", () => {
    expect(read("map.js").match(PRIVACY_JS_URL) ?? []).toEqual([]);
    expect(read("map.css").match(PRIVACY_CSS_URL) ?? []).toEqual([]);
    const fonts = read("fonts.css").match(PRIVACY_CSS_URL) ?? [];
    expect(fonts).toHaveLength(1);
    expect(fonts[0]).toContain("fonts.googleapis.com");
  });
});
