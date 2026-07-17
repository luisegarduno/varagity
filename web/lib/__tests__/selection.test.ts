import { describe, expect, it } from "vitest";

import type { DocumentOut } from "@/lib/api";
import {
  pruneSelection,
  selectedDocuments,
  selectionState,
  toggleSelected,
  totalChunks,
} from "@/lib/selection";

function makeDocument(docId: string, chunks = 3): DocumentOut {
  return {
    doc_id: docId,
    file_name: `${docId}.md`,
    source: `/docs/${docId}.md`,
    file_type: "md",
    content_hash: `hash-${docId}`,
    n_chunks: chunks,
    ingested_at: "2026-07-16T00:00:00Z",
    extraction_mix: { text: chunks },
  };
}

describe("selectionState", () => {
  it("is none when nothing is ticked", () => {
    expect(selectionState(0, 3)).toBe("none");
  });

  it("is some for a partial tick", () => {
    expect(selectionState(2, 3)).toBe("some");
  });

  it("is all when every row is ticked", () => {
    expect(selectionState(3, 3)).toBe("all");
  });

  it("is none for an empty table — 'all of nothing' would render ticked", () => {
    expect(selectionState(0, 0)).toBe("none");
  });
});

describe("toggleSelected", () => {
  it("adds an unticked id", () => {
    expect([...toggleSelected(new Set(["a"]), "b")]).toEqual(["a", "b"]);
  });

  it("removes a ticked id", () => {
    expect([...toggleSelected(new Set(["a", "b"]), "a")]).toEqual(["b"]);
  });

  it("does not mutate the input", () => {
    const before = new Set(["a"]);
    toggleSelected(before, "b");
    expect([...before]).toEqual(["a"]);
  });
});

describe("pruneSelection", () => {
  it("drops ids whose rows are gone", () => {
    const documents = [makeDocument("a")];
    expect([...pruneSelection(new Set(["a", "ghost"]), documents)]).toEqual([
      "a",
    ]);
  });

  it("keeps the same set instance when nothing is stale", () => {
    const selected = new Set(["a"]);
    expect(pruneSelection(selected, [makeDocument("a")])).toBe(selected);
  });

  it("leaves the selection alone while the table is loading", () => {
    const selected = new Set(["a"]);
    // null is "not loaded yet", not "no documents" — pruning here would
    // silently clear the selection on every refetch.
    expect(pruneSelection(selected, null)).toBe(selected);
  });

  it("empties against a table that came back empty", () => {
    expect([...pruneSelection(new Set(["a"]), [])]).toEqual([]);
  });
});

describe("selectedDocuments", () => {
  it("returns table order, not click order", () => {
    const documents = [makeDocument("a"), makeDocument("b"), makeDocument("c")];
    const chosen = selectedDocuments(documents, new Set(["c", "a"]));
    expect(chosen.map((document) => document.doc_id)).toEqual(["a", "c"]);
  });

  it("ignores ids with no row", () => {
    expect(selectedDocuments([makeDocument("a")], new Set(["ghost"]))).toEqual(
      [],
    );
  });

  it("is empty while loading", () => {
    expect(selectedDocuments(null, new Set(["a"]))).toEqual([]);
  });
});

describe("totalChunks", () => {
  it("sums the chunks a delete would remove", () => {
    expect(totalChunks([makeDocument("a", 2), makeDocument("b", 5)])).toBe(7);
  });

  it("is zero for nothing selected", () => {
    expect(totalChunks([])).toBe(0);
  });
});
