import { describe, expect, it } from "vitest";

import type { DocumentOut } from "@/lib/api";
import {
  buildCorpusTree,
  folderDocuments,
  latestIngestedAt,
  treeDocuments,
  visibleRows,
  type CorpusFolder,
  type CorpusTree,
} from "@/lib/corpus-tree";

function makeDocument(
  docId: string,
  relativePath: string | null,
  ingestedAt = "2026-07-16T00:00:00Z",
): DocumentOut {
  const name = relativePath?.split("/").at(-1) ?? `${docId}.md`;
  return {
    doc_id: docId,
    file_name: name,
    source: `/docs/${relativePath ?? name}`,
    relative_path: relativePath,
    file_type: "md",
    content_hash: `hash-${docId}`,
    n_chunks: 3,
    ingested_at: ingestedAt,
    extraction_mix: { text: 3 },
  };
}

function folderPaths(folders: readonly CorpusFolder[]): string[] {
  return folders.map((folder) => folder.path);
}

function rowKeys(tree: CorpusTree, expanded: Iterable<string>): string[] {
  return visibleRows(tree, new Set(expanded)).map((row) =>
    row.kind === "folder"
      ? `${"  ".repeat(row.depth)}${row.folder.name}/`
      : `${"  ".repeat(row.depth)}${row.document.doc_id}`,
  );
}

describe("buildCorpusTree", () => {
  it("keeps a flat corpus flat — no folders, order preserved", () => {
    const documents = [makeDocument("b", "b.md"), makeDocument("a", "a.md")];
    const tree = buildCorpusTree(documents);
    expect(tree.folders).toEqual([]);
    expect(tree.documents.map((d) => d.doc_id)).toEqual(["b", "a"]);
  });

  it("groups by the directory part and materializes ancestor folders", () => {
    const tree = buildCorpusTree([
      makeDocument("deep", "reports/2026/q3.md"),
      makeDocument("shallow", "reports/index.md"),
      makeDocument("root", "root.md"),
    ]);
    expect(folderPaths(tree.folders)).toEqual(["reports"]);
    const reports = tree.folders[0];
    expect(folderPaths(reports.folders)).toEqual(["reports/2026"]);
    expect(reports.documents.map((d) => d.doc_id)).toEqual(["shallow"]);
    expect(reports.folders[0].documents.map((d) => d.doc_id)).toEqual(["deep"]);
    expect(tree.documents.map((d) => d.doc_id)).toEqual(["root"]);
  });

  it("treats a missing relative_path as a root file", () => {
    const tree = buildCorpusTree([makeDocument("outside", null)]);
    expect(tree.folders).toEqual([]);
    expect(tree.documents.map((d) => d.doc_id)).toEqual(["outside"]);
  });

  it("never materializes an empty folder name from a leading slash", () => {
    const tree = buildCorpusTree([makeDocument("odd", "/oddball.md")]);
    expect(tree.folders).toEqual([]);
    expect(tree.documents.map((d) => d.doc_id)).toEqual(["odd"]);
  });

  it("natural-sorts folders at every level", () => {
    const tree = buildCorpusTree([
      makeDocument("c", "q10/c.md"),
      makeDocument("a", "q2/a.md"),
      makeDocument("b", "archive/b.md"),
    ]);
    expect(folderPaths(tree.folders)).toEqual(["archive", "q2", "q10"]);
  });

  it("keeps the given (newest-first) order for files within a folder", () => {
    const tree = buildCorpusTree([
      makeDocument("newest", "reports/new.md"),
      makeDocument("older", "reports/old.md"),
    ]);
    expect(tree.folders[0].documents.map((d) => d.doc_id)).toEqual([
      "newest",
      "older",
    ]);
  });
});

describe("visibleRows", () => {
  const tree = buildCorpusTree([
    makeDocument("deep", "reports/2026/q3.md"),
    makeDocument("shallow", "reports/index.md"),
    makeDocument("lone", "archive/a.md"),
    makeDocument("root", "root.md"),
  ]);

  it("shows only top-level folders and root files when nothing is expanded", () => {
    expect(rowKeys(tree, [])).toEqual(["archive/", "reports/", "root"]);
  });

  it("expands one folder without leaking its collapsed subfolders' children", () => {
    expect(rowKeys(tree, ["reports"])).toEqual([
      "archive/",
      "reports/",
      "  2026/",
      "  shallow",
      "root",
    ]);
  });

  it("shows grandchildren only when the whole chain is expanded", () => {
    expect(rowKeys(tree, ["reports", "reports/2026"])).toEqual([
      "archive/",
      "reports/",
      "  2026/",
      "    deep",
      "  shallow",
      "root",
    ]);
  });

  it("ignores expanded paths that no longer exist", () => {
    expect(rowKeys(tree, ["ghost"])).toEqual(["archive/", "reports/", "root"]);
  });
});

describe("folderDocuments / treeDocuments", () => {
  const tree = buildCorpusTree([
    makeDocument("deep", "reports/2026/q3.md"),
    makeDocument("shallow", "reports/index.md"),
    makeDocument("root", "root.md"),
  ]);

  it("collects every descendant in render order (subfolders first)", () => {
    expect(folderDocuments(tree.folders[0]).map((d) => d.doc_id)).toEqual([
      "deep",
      "shallow",
    ]);
  });

  it("walks the whole tree in render order", () => {
    expect(treeDocuments(tree).map((d) => d.doc_id)).toEqual([
      "deep",
      "shallow",
      "root",
    ]);
  });
});

describe("latestIngestedAt", () => {
  it("returns the most recent timestamp", () => {
    const documents = [
      makeDocument("old", "a.md", "2026-07-01T00:00:00Z"),
      makeDocument("new", "b.md", "2026-07-15T12:00:00Z"),
      makeDocument("mid", "c.md", "2026-07-10T00:00:00Z"),
    ];
    expect(latestIngestedAt(documents)).toBe("2026-07-15T12:00:00Z");
  });

  it("is null for an empty list", () => {
    expect(latestIngestedAt([])).toBeNull();
  });
});
