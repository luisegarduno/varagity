/**
 * The checked-in, precomputed map layout — the ELK pass over
 * `codebase-map.data.ts` runs at authoring time and ships as
 * `codebase-map.layout.json`, so `/map` hydrates instantly and elkjs never
 * loads in the browser (the drift-guarded `openapi.json` pattern).
 *
 * `__tests__/map-layout.test.ts` fails CI when the snapshot drifts from a
 * live ELK run; regenerate with `bun run test -u`.
 */

import {
  deserializeLayout,
  type MapLayout,
  type SerializedMapLayout,
} from "./map-layout";
import snapshot from "./codebase-map.layout.json";

/** Rehydrate the checked-in layout snapshot. */
export function precomputedLayout(): MapLayout {
  // The JSON module's literals widen (kinds become `string`), so the cast
  // goes through `unknown`; the snapshot is machine-generated from typed
  // data and drift-guarded, never hand-edited.
  return deserializeLayout(snapshot as unknown as SerializedMapLayout);
}
