import { describe, expect, it } from "vitest";

import { filesFromDrop } from "@/lib/dropped-files";

/** A `FileSystemFileEntry` fake; `readable: false` fails the `file()` call. */
function fileEntry(name: string, readable = true) {
  return {
    isFile: true,
    isDirectory: false,
    name,
    file: (ok: (file: File) => void, fail: (error: unknown) => void) =>
      readable ? ok(new File([new Uint8Array(8)], name)) : fail(new Error("gone")),
  };
}

/**
 * A `FileSystemDirectoryEntry` fake whose reader hands back at most 100
 * entries per call and ends with an empty batch — Chrome's actual contract,
 * and the thing a single `readEntries` call silently truncates.
 */
function dirEntry(name: string, children: unknown[]) {
  return {
    isFile: false,
    isDirectory: true,
    name,
    createReader: () => {
      let cursor = 0;
      return {
        readEntries: (ok: (entries: unknown[]) => void) => {
          const batch = children.slice(cursor, cursor + 100);
          cursor += batch.length;
          ok(batch);
        },
      };
    },
  };
}

/** A drop carrying `entries` (via the entries API) and `files` (the flat list). */
function drop(entries: unknown[], files: File[] = []): DataTransfer {
  return {
    files,
    items: entries.map((entry) => ({ kind: "file", webkitGetAsEntry: () => entry })),
  } as unknown as DataTransfer;
}

const paths = (files: File[]) => files.map((file) => file.webkitRelativePath);

describe("filesFromDrop", () => {
  it("descends into a dropped folder, keeping its structure", async () => {
    const dropped = await filesFromDrop(
      drop([
        dirEntry("mydocs", [
          fileEntry("a.pdf"),
          dirEntry("nested", [fileEntry("b.md")]),
        ]),
      ]),
    );

    expect(dropped.folder).toBe(true);
    expect(paths(dropped.files)).toEqual(["mydocs/a.pdf", "mydocs/nested/b.md"]);
    expect(dropped.files.map((file) => file.name)).toEqual(["a.pdf", "b.md"]);
  });

  it("reads every batch — a folder past the 100-entry cap is not truncated", async () => {
    const children = Array.from({ length: 250 }, (_, i) => fileEntry(`doc-${i}.md`));

    const dropped = await filesFromDrop(drop([dirEntry("big", children)]));

    expect(dropped.files).toHaveLength(250);
    expect(paths(dropped.files).at(-1)).toBe("big/doc-249.md");
  });

  it("takes every entry handle before its first await (the list is neutered after)", async () => {
    let items: unknown[] = [
      { kind: "file", webkitGetAsEntry: () => dirEntry("docs", [fileEntry("a.md")]) },
    ];
    const transfer = {
      files: [],
      get items() {
        return items;
      },
    } as unknown as DataTransfer;

    const walking = filesFromDrop(transfer); // not awaited — the handler returns
    items = []; // ...and the browser empties the list

    expect(paths((await walking).files)).toEqual(["docs/a.md"]);
  });

  it("passes a flat drop through untouched — no walk, no paths", async () => {
    const loose = [new File(["x"], "a.md"), new File(["y"], "b.pdf")];

    const dropped = await filesFromDrop(drop([fileEntry("a.md"), fileEntry("b.pdf")], loose));

    expect(dropped.folder).toBe(false);
    expect(dropped.files).toEqual(loose);
    expect(dropped.skipped).toEqual({});
  });

  it("falls back to the flat list when the entries API is unavailable", async () => {
    const loose = [new File(["x"], "a.md")];
    const transfer = {
      files: loose,
      items: [{ kind: "file", webkitGetAsEntry: () => null }],
    } as unknown as DataTransfer;

    const dropped = await filesFromDrop(transfer);

    expect(dropped.folder).toBe(false);
    expect(dropped.files).toEqual(loose);
  });

  it("roots loose files of a mixed drop at the corpus top level", async () => {
    const dropped = await filesFromDrop(
      drop([fileEntry("loose.md"), dirEntry("mydocs", [fileEntry("a.pdf")])]),
    );

    expect(dropped.folder).toBe(true);
    expect(paths(dropped.files)).toEqual(["loose.md", "mydocs/a.pdf"]);
  });

  it("skips a file it cannot read rather than failing the drop", async () => {
    const dropped = await filesFromDrop(
      drop([dirEntry("mydocs", [fileEntry("gone.md", false), fileEntry("here.md")])]),
    );

    expect(paths(dropped.files)).toEqual(["mydocs/here.md"]);
  });

  it("ignores non-file drag items (dragged text, links)", async () => {
    const transfer = {
      files: [],
      items: [{ kind: "string", webkitGetAsEntry: () => null }],
    } as unknown as DataTransfer;

    expect((await filesFromDrop(transfer)).files).toEqual([]);
  });

  it("prunes a pathological tree instead of hanging the tab", async () => {
    // A directory that is its own child — the loop a symlink can create.
    const loop: Record<string, unknown> = { isFile: false, isDirectory: true, name: "d" };
    loop.createReader = () => {
      let done = false;
      return {
        readEntries: (ok: (entries: unknown[]) => void) => {
          const batch = done ? [] : [loop];
          done = true;
          ok(batch);
        },
      };
    };

    const dropped = await filesFromDrop(drop([loop]));

    expect(dropped.files).toEqual([]);
    expect(dropped.skipped).toEqual({ "folder too deep": 1 });
  });
});
