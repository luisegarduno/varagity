"use client";

import { Trash2Icon } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
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
        <p className="animate-pulse text-xs text-muted-foreground">Loading…</p>
      ) : documents.length === 0 ? (
        <p className="text-xs text-muted-foreground">
          Nothing ingested yet — upload files above, then run an ingest.
        </p>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-border">
          <table className="w-full text-left text-sm">
            <thead className="bg-muted/50 text-xs text-muted-foreground">
              <tr>
                <th className="px-3 py-2 font-medium">File</th>
                <th className="px-3 py-2 font-medium">Type</th>
                <th className="px-3 py-2 font-medium">Chunks</th>
                <th className="px-3 py-2 font-medium">Extraction</th>
                <th className="px-3 py-2 font-medium">Ingested</th>
                <th className="px-3 py-2" />
              </tr>
            </thead>
            <tbody>
              {documents.map((document) => (
                <tr key={document.doc_id} className="border-t border-border">
                  <td className="max-w-64 truncate px-3 py-2" title={document.source}>
                    {document.file_name}
                  </td>
                  <td className="px-3 py-2">
                    <span className="rounded bg-accent px-1.5 py-0.5 font-mono text-xs uppercase">
                      {document.file_type}
                    </span>
                  </td>
                  <td className="px-3 py-2 tabular-nums">{document.n_chunks}</td>
                  <td className="px-3 py-2 text-xs text-muted-foreground">
                    {extractionMix(document)}
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

      <Dialog open={target !== null} onOpenChange={(open) => !open && setTarget(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete “{target?.file_name}”?</DialogTitle>
            <DialogDescription>
              Removes its {target?.n_chunks ?? 0} chunk(s) from both stores, so
              it can no longer ground answers. Persisted conversations keep
              their evidence snapshots.
            </DialogDescription>
          </DialogHeader>
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={removeFile}
              onChange={(event) => setRemoveFile(event.target.checked)}
            />
            Also remove the file from the corpus directory (otherwise the next
            ingest re-adds it)
          </label>
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          )}
          <DialogFooter showCloseButton>
            <Button variant="destructive" onClick={() => void confirmDelete()} disabled={busy}>
              {busy ? "Deleting…" : "Delete"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </section>
  );
}

/** Render the extraction mix compactly, e.g. `text ×3 · ocr ×1`. */
function extractionMix(document: DocumentOut): string {
  const entries = Object.entries(document.extraction_mix);
  if (entries.length === 0) return "—";
  return entries
    .map(([method, count]) => `${method === "ocr_fallback" ? "ocr" : method} ×${count}`)
    .join(" · ");
}
