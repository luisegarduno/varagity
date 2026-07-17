/**
 * Multi-select bookkeeping for the corpus table (spec_v2 §4.2).
 *
 * The interesting part isn't the ticking — it's what happens when the table
 * changes *under* a live selection. A delete, an ingest, or another tab's
 * edit refetches the document list, and ids that no longer exist must not
 * linger: not in the "N selected" count, not in the header checkbox's
 * tri-state, and above all not in the next delete's payload. So selection
 * is stored as ids but always read back through {@link pruneSelection},
 * which is the seam this module exists to make testable.
 */

import type { DocumentOut } from "@/lib/api";

/** The header checkbox's tri-state: nothing, part, or all of the table. */
export type SelectionState = "none" | "some" | "all";

/**
 * Classify a selection for the header checkbox. An empty table is `"none"`
 * — "all of nothing" would render a ticked box over no rows.
 */
export function selectionState(
  selectedCount: number,
  total: number,
): SelectionState {
  if (total === 0 || selectedCount === 0) return "none";
  return selectedCount >= total ? "all" : "some";
}

/** Add or remove one id, returning a new set (never mutating the input). */
export function toggleSelected(
  selected: ReadonlySet<string>,
  docId: string,
): ReadonlySet<string> {
  const next = new Set(selected);
  if (!next.delete(docId)) next.add(docId);
  return next;
}

/**
 * Drop selected ids that are no longer in the table, so a selection made
 * before a refetch can't outlive the rows it referred to.
 *
 * Returns the *same* set when nothing was stale — the common case by far,
 * and identity is what lets callers derive this on every render without
 * churning downstream memos. A `null` list means "still loading", which is
 * not evidence of absence, so the selection is left alone.
 */
export function pruneSelection(
  selected: ReadonlySet<string>,
  documents: readonly DocumentOut[] | null,
): ReadonlySet<string> {
  if (documents === null || selected.size === 0) return selected;
  const visible = new Set(documents.map((document) => document.doc_id));
  const live = [...selected].filter((docId) => visible.has(docId));
  return live.length === selected.size ? selected : new Set(live);
}

/**
 * The documents a selection refers to, in table order (not click order) —
 * the confirm dialog lists them, so they should read top-to-bottom the way
 * the table does.
 */
export function selectedDocuments(
  documents: readonly DocumentOut[] | null,
  selected: ReadonlySet<string>,
): DocumentOut[] {
  if (documents === null) return [];
  return documents.filter((document) => selected.has(document.doc_id));
}

/** Total chunks a pending delete would remove from both stores. */
export function totalChunks(documents: readonly DocumentOut[]): number {
  return documents.reduce((sum, document) => sum + document.n_chunks, 0);
}
