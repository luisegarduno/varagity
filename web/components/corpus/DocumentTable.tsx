"use client";

import {
  ChevronRightIcon,
  FolderIcon,
  FolderOpenIcon,
  Trash2Icon,
} from "lucide-react";
import { useState } from "react";

import {
  AlertDialog,
  AlertDialogClose,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Skeleton } from "@/components/ui/skeleton";
import { ApiError, deleteDocuments, type DocumentOut } from "@/lib/api";
import {
  buildCorpusTree,
  folderDocuments,
  latestIngestedAt,
  treeDocuments,
  visibleRows,
  type CorpusFolder,
} from "@/lib/corpus-tree";
import {
  pruneSelection,
  selectedDocuments,
  selectionState,
  toggleSelected,
  totalChunks,
} from "@/lib/selection";
import { cn } from "@/lib/utils";

/**
 * The ingested-documents table (spec_v2 §4.2): file, type badge, chunk
 * count, ingested-at, extraction mix (how much came through OCR), and
 * delete — the GUI-driven GC for the v1 "removing a file doesn't remove its
 * chunks" gap. Delete removes from both stores; the confirm offers removing
 * the source files too so the next ingest can't resurrect them.
 *
 * Folder uploads keep their directory structure, and the table folds the
 * flat list back into it (`lib/corpus-tree.ts`): folders render as
 * collapsible rows with a descendant file count, summed chunks, and the
 * latest ingest date. A folder's checkbox and trash act on its descendants,
 * so "delete a directory" is just a selection of many.
 *
 * Rows are multi-selectable, and one row's trash button — file or folder —
 * is just a selection: everything goes through the same confirm and the
 * same bulk request, so there's a single delete path to reason about
 * rather than three that can drift.
 */
export function DocumentTable({
  documents,
  onChanged,
}: {
  documents: DocumentOut[] | null;
  onChanged: () => void;
}) {
  const [selected, setSelected] = useState<ReadonlySet<string>>(new Set());
  const [expanded, setExpanded] = useState<ReadonlySet<string>>(new Set());
  // Pending is the *committed* delete target: null while the dialog is
  // closed. Kept apart from `selected` so the confirm keeps naming what the
  // user agreed to even as the table refetches beneath it. `folder` records
  // that the user aimed at a directory, so the confirm can say so.
  const [pending, setPending] = useState<{
    targets: DocumentOut[];
    folder: string | null;
  } | null>(null);
  const [removeFile, setRemoveFile] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Derived every render rather than synced: a refetch (this delete, an
  // ingest, another tab) must not leave ids selected whose rows are gone,
  // and folders exist only as long as documents live beneath them.
  const tree = buildCorpusTree(documents ?? []);
  const ordered = treeDocuments(tree);
  const rows = visibleRows(tree, expanded);
  const hasFolders = tree.folders.length > 0;
  const live = pruneSelection(selected, documents);
  const chosen = selectedDocuments(ordered, live);
  const headerState = selectionState(live.size, documents?.length ?? 0);

  function openConfirm(targets: DocumentOut[], folder: string | null = null) {
    if (targets.length === 0) return;
    setPending({ targets, folder });
    setRemoveFile(true);
    setError(null);
  }

  function toggleExpanded(path: string) {
    setExpanded((current) => {
      const next = new Set(current);
      if (!next.delete(path)) next.add(path);
      return next;
    });
  }

  function toggleFolderSelection(folder: CorpusFolder) {
    const constituents = folderDocuments(folder);
    const allSelected = constituents.every((d) => live.has(d.doc_id));
    const next = new Set(live);
    for (const document of constituents) {
      if (allSelected) next.delete(document.doc_id);
      else next.add(document.doc_id);
    }
    setSelected(next);
  }

  async function confirmDelete() {
    if (pending === null || busy) return;
    setBusy(true);
    try {
      const doomed = pending.targets.map((document) => document.doc_id);
      await deleteDocuments(doomed, { removeFile });
      // Unselect exactly what was deleted; a row ticked while the request
      // was in flight stays ticked.
      setSelected((current) => {
        const next = new Set(current);
        for (const docId of doomed) next.delete(docId);
        return next;
      });
      setPending(null);
      setError(null);
      onChanged();
    } catch (failure) {
      setError(failure instanceof ApiError ? failure.message : String(failure));
    } finally {
      setBusy(false);
    }
  }

  const pendingTargets = pending?.targets ?? [];
  const pendingChunks = totalChunks(pendingTargets);

  return (
    <section aria-label="Ingested documents" className="flex flex-col gap-2">
      <div className="flex min-h-8 items-center justify-between gap-3">
        <h2 className="text-sm font-semibold">Documents</h2>
        {live.size > 0 && (
          <div className="flex items-center gap-3">
            <span aria-live="polite" className="text-xs text-muted-foreground">
              {live.size} selected
            </span>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setSelected(new Set())}
            >
              Clear
            </Button>
            <Button
              variant="destructive"
              size="sm"
              onClick={() => openConfirm(chosen)}
            >
              <Trash2Icon className="size-4" />
              Delete {live.size}
            </Button>
          </div>
        )}
      </div>

      {documents === null ? (
        <div className="overflow-hidden rounded-lg border border-border">
          {[0, 1, 2].map((row) => (
            <div
              key={row}
              className="flex items-center gap-4 border-t border-border px-3 py-3 first:border-t-0"
            >
              <Skeleton className="h-4 w-44" />
              <Skeleton className="h-4 w-10" />
              <Skeleton className="ml-auto h-4 w-28" />
            </div>
          ))}
        </div>
      ) : documents.length === 0 ? (
        <p className="text-xs text-muted-foreground">
          Nothing ingested yet — upload files above, then run an ingest.
        </p>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-border">
          <table className="w-full text-left text-sm">
            <thead className="bg-muted/40 text-xs text-muted-foreground">
              <tr>
                <th className="w-0 px-3 py-2">
                  <Checkbox
                    aria-label={
                      headerState === "all"
                        ? "Deselect all documents"
                        : "Select all documents"
                    }
                    checked={headerState === "all"}
                    indeterminate={headerState === "some"}
                    onCheckedChange={() =>
                      setSelected(
                        headerState === "all"
                          ? new Set()
                          : new Set(documents.map((d) => d.doc_id)),
                      )
                    }
                  />
                </th>
                <th className="px-3 py-2 font-medium">File</th>
                <th className="px-3 py-2 font-medium">Type</th>
                <th className="px-3 py-2 font-medium">Chunks</th>
                <th className="px-3 py-2 font-medium">Extraction</th>
                <th className="px-3 py-2 font-medium">Ingested</th>
                <th className="px-3 py-2">
                  <span className="sr-only">Actions</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) =>
                row.kind === "folder" ? (
                  <FolderRow
                    key={`folder:${row.folder.path}`}
                    folder={row.folder}
                    depth={row.depth}
                    open={expanded.has(row.folder.path)}
                    selected={live}
                    onToggleExpand={toggleExpanded}
                    onToggleSelect={toggleFolderSelection}
                    onDelete={(folder) =>
                      openConfirm(folderDocuments(folder), folder.path)
                    }
                  />
                ) : (
                  <tr
                    key={row.document.doc_id}
                    data-selected={live.has(row.document.doc_id) || undefined}
                    className="border-t border-border transition-colors hover:bg-muted/40 data-selected:bg-muted/60"
                  >
                    <td className="w-0 px-3 py-2">
                      <Checkbox
                        aria-label={`Select ${row.document.file_name}`}
                        checked={live.has(row.document.doc_id)}
                        onCheckedChange={() =>
                          setSelected(toggleSelected(live, row.document.doc_id))
                        }
                      />
                    </td>
                    <td
                      className="max-w-64 px-3 py-2"
                      title={row.document.source}
                    >
                      <span
                        className="flex min-w-0 items-center"
                        style={
                          // Sit file names under their (sibling) folder
                          // names; an all-flat corpus keeps the plain look.
                          hasFolders
                            ? {
                                paddingInlineStart: `${row.depth * 1.25 + 2.75}rem`,
                              }
                            : undefined
                        }
                      >
                        <span className="truncate">
                          {row.document.file_name}
                        </span>
                      </span>
                    </td>
                    <td className="px-3 py-2">
                      <Badge variant="outline" className="font-mono uppercase">
                        {row.document.file_type}
                      </Badge>
                    </td>
                    <td className="px-3 py-2 font-mono text-xs tabular-nums">
                      {row.document.n_chunks}
                    </td>
                    <td className="px-3 py-2">
                      <ExtractionMix document={row.document} />
                    </td>
                    <td
                      className="px-3 py-2 text-xs text-muted-foreground"
                      title={row.document.ingested_at}
                    >
                      {new Date(row.document.ingested_at).toLocaleString()}
                    </td>
                    <td className="px-3 py-2 text-right">
                      <Button
                        variant="ghost"
                        size="icon-sm"
                        aria-label={`Delete ${row.document.file_name}`}
                        onClick={() => openConfirm([row.document])}
                      >
                        <Trash2Icon className="size-4" />
                      </Button>
                    </td>
                  </tr>
                ),
              )}
            </tbody>
          </table>
        </div>
      )}

      <AlertDialog
        open={pending !== null}
        onOpenChange={(open) => {
          if (!open) setPending(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {pending?.folder != null
                ? `Delete folder “${pending.folder}”?`
                : pendingTargets.length === 1
                  ? `Delete “${pendingTargets[0].file_name}”?`
                  : `Delete ${pendingTargets.length} documents?`}
            </AlertDialogTitle>
            <AlertDialogDescription>
              Removes {pendingTargets.length === 1 ? "its" : "their"}{" "}
              {pendingChunks} chunk(s) from both stores, so{" "}
              {pendingTargets.length === 1 ? "it" : "they"} can no longer
              ground answers. Persisted conversations keep their evidence
              snapshots.
            </AlertDialogDescription>
          </AlertDialogHeader>
          {pendingTargets.length > 1 && (
            <ul className="max-h-40 overflow-y-auto rounded-md border border-border bg-muted/30 px-3 py-2 text-xs">
              {pendingTargets.map((document) => (
                <li
                  key={document.doc_id}
                  className="truncate py-0.5"
                  title={document.source}
                >
                  {document.relative_path ?? document.file_name}
                </li>
              ))}
            </ul>
          )}
          <label className="flex items-start gap-2 text-sm">
            <input
              type="checkbox"
              className="mt-0.5 size-4 accent-primary"
              checked={removeFile}
              onChange={(event) => setRemoveFile(event.target.checked)}
            />
            <span>
              Also remove{" "}
              {pendingTargets.length === 1 ? "the file" : "the files"} from the
              corpus directory (otherwise the next ingest re-adds{" "}
              {pendingTargets.length === 1 ? "it" : "them"})
            </span>
          </label>
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          )}
          <AlertDialogFooter>
            <AlertDialogClose render={<Button variant="outline" />}>
              Cancel
            </AlertDialogClose>
            <Button
              variant="destructive"
              onClick={() => void confirmDelete()}
              disabled={busy}
            >
              {busy ? "Deleting…" : "Delete"}
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </section>
  );
}

/**
 * One collapsible folder row. The checkbox and trash act on every
 * descendant document (subfolders included); the chunk, extraction, and
 * ingested cells aggregate the same set, so a collapsed folder still says
 * what it holds.
 */
function FolderRow({
  folder,
  depth,
  open,
  selected,
  onToggleExpand,
  onToggleSelect,
  onDelete,
}: {
  folder: CorpusFolder;
  depth: number;
  open: boolean;
  selected: ReadonlySet<string>;
  onToggleExpand: (path: string) => void;
  onToggleSelect: (folder: CorpusFolder) => void;
  onDelete: (folder: CorpusFolder) => void;
}) {
  const constituents = folderDocuments(folder);
  const ticked = constituents.filter((d) => selected.has(d.doc_id)).length;
  const state = selectionState(ticked, constituents.length);
  const latest = latestIngestedAt(constituents);
  return (
    <tr
      data-folder-path={folder.path}
      data-selected={state === "all" || undefined}
      className="border-t border-border transition-colors hover:bg-muted/40 data-selected:bg-muted/60"
    >
      <td className="w-0 px-3 py-2">
        <Checkbox
          aria-label={`Select folder ${folder.path}`}
          checked={state === "all"}
          indeterminate={state === "some"}
          onCheckedChange={() => onToggleSelect(folder)}
        />
      </td>
      <td className="max-w-64 px-3 py-2">
        <button
          type="button"
          aria-expanded={open}
          className="flex w-full min-w-0 items-center gap-1.5 text-left"
          style={{ paddingInlineStart: `${depth * 1.25}rem` }}
          onClick={() => onToggleExpand(folder.path)}
        >
          <ChevronRightIcon
            aria-hidden
            className={cn(
              "size-4 shrink-0 text-muted-foreground transition-transform",
              open && "rotate-90",
            )}
          />
          {open ? (
            <FolderOpenIcon
              aria-hidden
              className="size-4 shrink-0 text-muted-foreground"
            />
          ) : (
            <FolderIcon
              aria-hidden
              className="size-4 shrink-0 text-muted-foreground"
            />
          )}
          <span className="truncate font-medium">{folder.name}</span>
          <span className="shrink-0 text-xs text-muted-foreground">
            {constituents.length} file{constituents.length === 1 ? "" : "s"}
          </span>
        </button>
      </td>
      <td className="px-3 py-2 text-xs text-muted-foreground">—</td>
      <td className="px-3 py-2 font-mono text-xs tabular-nums">
        {totalChunks(constituents)}
      </td>
      <td className="px-3 py-2 text-xs text-muted-foreground">—</td>
      <td
        className="px-3 py-2 text-xs text-muted-foreground"
        title={latest ?? undefined}
      >
        {latest === null ? "—" : new Date(latest).toLocaleString()}
      </td>
      <td className="px-3 py-2 text-right">
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label={`Delete folder ${folder.path}`}
          onClick={() => onDelete(folder)}
        >
          <Trash2Icon className="size-4" />
        </Button>
      </td>
    </tr>
  );
}

/**
 * Render the extraction mix: plain-text methods stay quiet mono text, any
 * OCR-fallback chunks get a warning badge (the lower-fidelity path).
 */
function ExtractionMix({ document }: { document: DocumentOut }) {
  const entries = Object.entries(document.extraction_mix);
  if (entries.length === 0) {
    return <span className="text-xs text-muted-foreground">—</span>;
  }
  const ocrCount = document.extraction_mix.ocr_fallback ?? 0;
  return (
    <span className="flex flex-wrap items-center gap-1.5">
      {entries
        .filter(([method]) => method !== "ocr_fallback")
        .map(([method, count]) => (
          <span key={method} className="font-mono text-xs text-muted-foreground">
            {method} ×{count}
          </span>
        ))}
      {ocrCount > 0 && (
        <Badge variant="warning" className="font-mono">
          ocr ×{ocrCount}
        </Badge>
      )}
    </span>
  );
}
