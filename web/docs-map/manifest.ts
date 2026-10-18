/**
 * The docs-site map bundle's drift manifest (ADR-018).
 *
 * `golden-docs/assets/map/` is a checked-in build of this directory — the
 * `openapi.json` pattern: a generated artifact beside the docs, regenerated
 * by hand (`bun run build:docs-map`) and guarded by a test that fails CI
 * when it is stale. Minified output is not byte-stable across tool versions,
 * so the guard compares a hash of the bundle's *inputs* instead: the map
 * components, data, layout snapshot, styles, icons, and this directory.
 * `vite.config.ts` writes the manifest; `lib/__tests__/docs-map.test.ts`
 * recomputes the hash and compares.
 */

import { createHash } from "node:crypto";
import { readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";

/**
 * Files and directories, relative to `web/`, whose content shapes the
 * bundle. Dependency versions are deliberately not among them: a React
 * patch changes the bytes but not the picture the docs reader sees, and
 * the committed bundle keeps working regardless.
 */
export const DOCS_MAP_INPUTS: readonly string[] = [
  "app/globals.css",
  "components/map",
  "components/ui/button.tsx",
  "docs-map",
  "lib/codebase-map.data.ts",
  "lib/codebase-map.layout.json",
  "lib/codebase-map.layout.ts",
  "lib/codebase-map.ts",
  "lib/map-layout.ts",
  "lib/utils.ts",
  "public/map-icons",
];

/** Files the build emits, relative to its output directory (icons aside). */
export const DOCS_MAP_OUTPUTS: readonly string[] = [
  "index.html",
  "map.js",
  "map.css",
  "fonts.css",
  "manifest.json",
];

/** What `manifest.json` records. */
export interface DocsMapManifest {
  /** How to regenerate — for the human reading a failed guard. */
  regenerate: string;
  /** `sha256:` over every input file's path and bytes, in sorted order. */
  sourceHash: string;
  /** The files hashed, relative to `web/`, sorted. */
  inputs: string[];
}

/** The regeneration command, as the guard's failure message quotes it. */
export const REGENERATE_COMMAND = "bun run build:docs-map";

/**
 * Expand {@link DOCS_MAP_INPUTS} into the sorted list of files they cover,
 * as forward-slash paths relative to `webRoot` (so the hash is the same on
 * every platform).
 */
export function listInputFiles(webRoot: string): string[] {
  const files: string[] = [];
  const visit = (abs: string): void => {
    if (statSync(abs).isDirectory()) {
      for (const name of readdirSync(abs)) visit(path.join(abs, name));
    } else {
      files.push(path.relative(webRoot, abs).split(path.sep).join("/"));
    }
  };
  for (const input of DOCS_MAP_INPUTS) visit(path.join(webRoot, input));
  return files.sort();
}

/** One sha256 over each input file's relative path and bytes, in order. */
export function hashInputs(webRoot: string, files = listInputFiles(webRoot)): string {
  const hash = createHash("sha256");
  for (const file of files) {
    hash.update(file);
    hash.update("\0");
    hash.update(readFileSync(path.join(webRoot, file)));
    hash.update("\0");
  }
  return `sha256:${hash.digest("hex")}`;
}

/** The manifest for the inputs as they are on disk right now. */
export function buildManifest(webRoot: string): DocsMapManifest {
  const inputs = listInputFiles(webRoot);
  return {
    regenerate: REGENERATE_COMMAND,
    sourceHash: hashInputs(webRoot, inputs),
    inputs,
  };
}
