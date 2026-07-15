"use client";

import { UploadIcon } from "lucide-react";
import { useRef, useState } from "react";

import { Badge } from "@/components/ui/badge";
import {
  ApiError,
  uploadDocuments,
  type ConfigResponse,
  type UploadedFile,
} from "@/lib/api";
import { validateUpload } from "@/lib/upload";
import { cn } from "@/lib/utils";

interface UploadOutcome {
  fileName: string;
  ok: boolean;
  detail: string;
}

/** Human phrasing for the client-side rejection reasons (lib/upload.ts). */
const REJECTION_LABELS: Record<string, string> = {
  extension_not_allowed: "file type not allowed",
  file_too_large: "over the size limit",
  invalid_filename: "invalid file name",
};

/**
 * Drag-drop / click-to-pick upload into `DOCS_PATH` (spec_v2 §4.2).
 * Client-side validation mirrors the server's extension/size rules (from
 * `GET /api/config`) so bad files are refused before uploading; accepted
 * files go up in one multipart POST and the per-file outcomes render.
 * Uploading does **not** ingest — that's the explicit ingest action.
 */
export function UploadDropzone({
  config,
  onUploaded,
}: {
  config: ConfigResponse | null;
  onUploaded: () => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [outcomes, setOutcomes] = useState<UploadOutcome[]>([]);

  async function handleFiles(files: File[]) {
    if (files.length === 0 || busy) return;
    const rejected: UploadOutcome[] = [];
    const accepted: File[] = [];
    for (const file of files) {
      const check = config
        ? validateUpload(
            file.name,
            file.size,
            config.allowed_extensions,
            config.upload_max_mb,
          )
        : { fileName: file.name, ok: true as const };
      if (check.ok) accepted.push(file);
      else
        rejected.push({
          fileName: check.fileName,
          ok: false,
          detail:
            (check.reason && REJECTION_LABELS[check.reason]) ??
            check.reason ??
            "rejected",
        });
    }

    let stored: UploadOutcome[] = [];
    if (accepted.length > 0) {
      setBusy(true);
      try {
        const response = await uploadDocuments(accepted);
        stored = response.files.map((entry: UploadedFile) => ({
          fileName: entry.file_name,
          ok: entry.stored,
          detail: entry.stored
            ? entry.replaced
              ? "replaced — re-ingest to pick up the new content"
              : "uploaded — not yet ingested"
            : (entry.reason ?? "rejected"),
        }));
        if (response.files.some((entry) => entry.stored)) onUploaded();
      } catch (failure) {
        stored = [
          {
            fileName: `${accepted.length} file(s)`,
            ok: false,
            detail: failure instanceof ApiError ? failure.message : String(failure),
          },
        ];
      } finally {
        setBusy(false);
      }
    }
    setOutcomes([...stored, ...rejected]);
  }

  const constraints = config
    ? `${config.allowed_extensions.join(" ")} · up to ${config.upload_max_mb} MB each`
    : "";

  return (
    <section aria-label="Upload documents">
      <div
        role="button"
        tabIndex={0}
        aria-label="Upload documents into the corpus"
        onClick={() => inputRef.current?.click()}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            inputRef.current?.click();
          }
        }}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={(event) => {
          // Ignore leave events fired by moving over the dropzone's children.
          if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
            setDragging(false);
          }
        }}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          void handleFiles(Array.from(event.dataTransfer.files));
        }}
        className={cn(
          "flex cursor-pointer flex-col items-center gap-2 rounded-xl border border-dashed border-border p-8 text-center select-none motion-safe:transition-colors motion-safe:duration-150",
          dragging
            ? "border-ring bg-primary/5"
            : "hover:border-muted-foreground/40 hover:bg-muted/30",
          busy && "pointer-events-none opacity-60",
        )}
      >
        <span
          aria-hidden
          className="flex size-9 items-center justify-center rounded-full bg-muted"
        >
          <UploadIcon className="size-4 text-muted-foreground" />
        </span>
        <p className="text-sm font-medium">
          {busy ? "Uploading…" : "Drop files here or click to upload"}
        </p>
        <p className="text-xs text-muted-foreground">{constraints}</p>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={config?.allowed_extensions.join(",")}
          className="hidden"
          onChange={(event) => {
            void handleFiles(Array.from(event.target.files ?? []));
            event.target.value = ""; // allow re-picking the same file
          }}
        />
      </div>

      {outcomes.length > 0 && (
        <ul className="mt-2 flex flex-col gap-1" aria-label="Upload outcomes">
          {outcomes.map((outcome, index) => (
            <li
              key={`${outcome.fileName}-${index}`}
              className="flex flex-wrap items-center gap-2 rounded-md border border-border/60 px-2.5 py-1.5 text-xs"
            >
              <Badge variant={outcome.ok ? "success" : "destructive"}>
                {outcome.ok ? "uploaded" : "rejected"}
              </Badge>
              <span className="max-w-56 truncate font-medium" title={outcome.fileName}>
                {outcome.fileName}
              </span>
              <span className="text-muted-foreground">{outcome.detail}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
