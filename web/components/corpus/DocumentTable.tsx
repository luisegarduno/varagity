"use client";

import { Trash2Icon } from "lucide-react";
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
import { Skeleton } from "@/components/ui/skeleton";
import { ApiError, deleteDocument, type DocumentOut } from "@/lib/api";

/**
 * The ingested-documents table (spec_v2 §4.2): file, type badge, chunk
 * count, ingested-at, extraction mix (how much came through OCR), and the
 * per-document delete — the GUI-driven GC for the v1 "removing a file
 * doesn't remove its chunks" gap. Delete removes from both stores; the
 * confirm offers removing the source file too so the next ingest can't
 * resurrect it.
 */
export function DocumentTable({
  documents,
  onChanged,
}: {
  documents: DocumentOut[] | null;
  onChanged: () => void;
}) {
  const [target, setTarget] = useState<DocumentOut | null>(null);
  const [removeFile, setRemoveFile] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function confirmDelete() {
    if (target === null || busy) return;
    setBusy(true);
    try {
      await deleteDocument(target.doc_id, { removeFile });
      setTarget(null);
      setError(null);
      onChanged();
    } catch (failure) {
      setError(failure instanceof ApiError ? failure.message : String(failure));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section aria-label="Ingested documents" className="flex flex-col gap-2">
      <h2 className="text-sm font-semibold">Documents</h2>

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
              {documents.map((document) => (
                <tr
                  key={document.doc_id}
                  className="border-t border-border transition-colors hover:bg-muted/40"
                >
                  <td className="max-w-64 truncate px-3 py-2" title={document.source}>
                    {document.file_name}
                  </td>
                  <td className="px-3 py-2">
                    <Badge variant="outline" className="font-mono uppercase">
                      {document.file_type}
                    </Badge>
                  </td>
                  <td className="px-3 py-2 font-mono text-xs tabular-nums">
                    {document.n_chunks}
                  </td>
                  <td className="px-3 py-2">
                    <ExtractionMix document={document} />
                  </td>
                  <td
                    className="px-3 py-2 text-xs text-muted-foreground"
                    title={document.ingested_at}
                  >
                    {new Date(document.ingested_at).toLocaleString()}
                  </td>
                  <td className="px-3 py-2 text-right">
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      aria-label={`Delete ${document.file_name}`}
                      onClick={() => {
                        setTarget(document);
                        setRemoveFile(true);
                        setError(null);
                      }}
                    >
                      <Trash2Icon className="size-4" />
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <AlertDialog
        open={target !== null}
        onOpenChange={(open) => {
          if (!open) setTarget(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete “{target?.file_name}”?</AlertDialogTitle>
            <AlertDialogDescription>
              Removes its {target?.n_chunks ?? 0} chunk(s) from both stores, so
              it can no longer ground answers. Persisted conversations keep
              their evidence snapshots.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <label className="flex items-start gap-2 text-sm">
            <input
              type="checkbox"
              className="mt-0.5 size-4 accent-primary"
              checked={removeFile}
              onChange={(event) => setRemoveFile(event.target.checked)}
            />
            <span>
              Also remove the file from the corpus directory (otherwise the next
              ingest re-adds it)
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
