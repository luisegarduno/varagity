/**
 * Builds the docs-site map bundle (ADR-018): `bun run build:docs-map`.
 *
 * Vite runs under Node (ADR-014: bun is the package manager only) and emits
 * one page, one script, one stylesheet, the vendored favicons, the fonts
 * stylesheet, and the drift manifest into `golden-docs/assets/map/`, which
 * MkDocs copies verbatim and `golden-docs/map.md` frames. `base: "./"`
 * keeps every URL relative — GitHub Pages serves the site under `/varagity/`.
 * Tailwind runs through its Vite plugin here, so the app's PostCSS config is
 * explicitly not loaded (it would run Tailwind a second time).
 */

import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";

import tailwindcss from "@tailwindcss/vite";
import { defineConfig, type Plugin } from "vite";

import { buildManifest } from "./manifest";

const WEB_ROOT = path.resolve(import.meta.dirname, "..");
const OUT_DIR = path.resolve(WEB_ROOT, "../golden-docs/assets/map");
const ICONS_DIR = path.join(WEB_ROOT, "public/map-icons");

/**
 * What the bundler can't know about: the favicons `icon-base.ts` resolves
 * beside the page, `fonts.css` (copied verbatim — see its comment), the
 * drift manifest, and the `<link>` that loads the fonts (injected rather
 * than written in index.html so Vite doesn't bundle it away).
 */
function docsMapAssets(): Plugin {
  return {
    name: "varagity:docs-map-assets",
    transformIndexHtml() {
      return [
        {
          tag: "link",
          attrs: { rel: "stylesheet", href: "fonts.css" },
          injectTo: "head",
        },
      ];
    },
    generateBundle() {
      for (const name of readdirSync(ICONS_DIR)) {
        if (!name.endsWith(".png")) continue;
        this.emitFile({
          type: "asset",
          fileName: `map-icons/${name}`,
          source: readFileSync(path.join(ICONS_DIR, name)),
        });
      }
      this.emitFile({
        type: "asset",
        fileName: "fonts.css",
        source: readFileSync(path.join(import.meta.dirname, "fonts.css")),
      });
      this.emitFile({
        type: "asset",
        fileName: "manifest.json",
        source: `${JSON.stringify(buildManifest(WEB_ROOT), null, 2)}\n`,
      });
    },
  };
}

export default defineConfig({
  root: import.meta.dirname,
  base: "./",
  publicDir: false,
  resolve: { alias: { "@": WEB_ROOT } },
  // An inline (empty) PostCSS config stops Vite from discovering
  // web/postcss.config.mjs, whose @tailwindcss/postcss would double up.
  css: { postcss: {} },
  plugins: [tailwindcss(), docsMapAssets()],
  build: {
    outDir: OUT_DIR,
    emptyOutDir: true,
    sourcemap: false,
    modulePreload: { polyfill: false },
    rolldownOptions: {
      output: {
        entryFileNames: "map.js",
        chunkFileNames: "[name].js",
        assetFileNames: (asset) =>
          (asset.names[0] ?? "").endsWith(".css") ? "map.css" : "[name][extname]",
      },
    },
  },
});
