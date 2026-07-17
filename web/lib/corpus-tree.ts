/**
 * Folder grouping for the corpus table (the read side of folder uploads,
 * spec_v3 §5.2).
 *
 * Folder uploads preserve directory structure on disk, and `GET
 * /api/documents` reports each document's `relative_path` under the corpus
 * root. This module folds that flat list back into the folder tree the
 * upload came from — pure data work, kept out of the component so the
 * grouping, ordering, and expansion rules stay testable without a DOM.
 *
 * Ordering is the file-explorer convention: folders first (natural-sorted,
 * so "q10" follows "q2"), then files in the list's given order (the API's
 * newest-ingest-first). Documents without a usable `relative_path`
 * (ingested from outside `DOCS_PATH`) list flat at the root, exactly as
 * the table has always rendered them.
 */

import type { DocumentOut } from "@/lib/api";

/** One folder node — direct children only; descendants via {@link folderDocuments}. */
export interface CorpusFolder {
  /** The folder's own name (the path's last segment). */
  name: string;
  /** Full path from the corpus root (`"reports/2026"`) — the stable key. */
  path: string;
  folders: CorpusFolder[];
  documents: DocumentOut[];
}

/** The corpus root: top-level folders plus the files living at the root. */
export interface CorpusTree {
  folders: CorpusFolder[];
  documents: DocumentOut[];
}

/** One renderable table row: a folder or a file, at its indentation depth. */
export type CorpusRow =
  | { kind: "folder"; folder: CorpusFolder; depth: number }
  | { kind: "file"; document: DocumentOut; depth: number };

function byName(a: CorpusFolder, b: CorpusFolder): number {
  return a.name.localeCompare(b.name, undefined, {
    numeric: true,
    sensitivity: "base",
  });
}

function sortFolders(folders: CorpusFolder[]): void {
  folders.sort(byName);
  for (const folder of folders) sortFolders(folder.folders);
}

/**
 * Fold the flat document list into its folder tree.
 *
 * The grouping key is the directory part of `relative_path`; a document
 * with no `relative_path`, or one without a directory part, lands at the
 * root. Every ancestor folder along a path materializes, so
 * `"reports/2026/q3.pdf"` yields `reports` holding `2026` — folders exist
 * only as long as they hold documents somewhere beneath them.
 */
export function buildCorpusTree(documents: readonly DocumentOut[]): CorpusTree {
  const root: CorpusTree = { folders: [], documents: [] };
  const folderIndex = new Map<string, CorpusFolder>();

  function folderFor(path: string): CorpusFolder {
    const existing = folderIndex.get(path);
    if (existing !== undefined) return existing;
    const cut = path.lastIndexOf("/");
    const parent = cut === -1 ? root : folderFor(path.slice(0, cut));
    const folder: CorpusFolder = {
      name: path.slice(cut + 1),
      path,
      folders: [],
      documents: [],
    };
    parent.folders.push(folder);
    folderIndex.set(path, folder);
    return folder;
  }

  for (const document of documents) {
    const relative = document.relative_path ?? null;
    // `cut > 0` (not `!== -1`): a path can't legitimately start with "/",
    // but an empty folder name must never materialize from one that does.
    const cut = relative === null ? -1 : relative.lastIndexOf("/");
    const target =
      relative !== null && cut > 0 ? folderFor(relative.slice(0, cut)) : root;
    target.documents.push(document);
  }

  sortFolders(root.folders);
  return root;
}

/**
 * Flatten the tree into the rows the table renders, honoring which folder
 * paths are expanded. Children of a collapsed folder (files and subfolders
 * alike) don't appear; a subfolder's children need their own expansion.
 */
export function visibleRows(
  tree: CorpusTree,
  expanded: ReadonlySet<string>,
): CorpusRow[] {
  const rows: CorpusRow[] = [];
  function walk(
    folders: readonly CorpusFolder[],
    documents: readonly DocumentOut[],
    depth: number,
  ): void {
    for (const folder of folders) {
      rows.push({ kind: "folder", folder, depth });
      if (expanded.has(folder.path)) {
        walk(folder.folders, folder.documents, depth + 1);
      }
    }
    for (const document of documents) {
      rows.push({ kind: "file", document, depth });
    }
  }
  walk(tree.folders, tree.documents, 0);
  return rows;
}

/**
 * Every document under a folder, in render order (subfolders' documents
 * first, then the folder's own) — the folder checkbox's constituency and
 * the folder delete's payload.
 */
export function folderDocuments(folder: CorpusFolder): DocumentOut[] {
  return [...folder.folders.flatMap(folderDocuments), ...folder.documents];
}

/**
 * Every document in the tree, in render order — what "select all" selects,
 * and the order the confirm dialog lists a mixed selection in (top-to-
 * bottom the way the table reads).
 */
export function treeDocuments(tree: CorpusTree): DocumentOut[] {
  return [...tree.folders.flatMap(folderDocuments), ...tree.documents];
}

/**
 * The most recent `ingested_at` among documents — a folder row's
 * "Ingested" cell. `null` for an empty list (never rendered, but total).
 */
export function latestIngestedAt(
  documents: readonly DocumentOut[],
): string | null {
  let latest: string | null = null;
  for (const document of documents) {
    if (latest === null || Date.parse(document.ingested_at) > Date.parse(latest)) {
      latest = document.ingested_at;
    }
  }
  return latest;
}
