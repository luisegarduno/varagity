"use client";

import { UploadIcon } from "lucide-react";
import { useRef, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { type ConfigResponse } from "@/lib/api";
import { useUpload } from "@/lib/use-upload";
import { cn } from "@/lib/utils";

/**
 * Drag-drop / click-to-pick upload into `DOCS_PATH` (spec_v2 §4.2).
 * The validate → upload → report machinery lives in the `useUpload` hook
 * (spec_v3 §5.3); this component owns only the dropzone chrome, the
 * per-file outcome rows, and a folder drop's one-line summary.
 * Uploading does **not** ingest — that's the explicit ingest action.
 *
 * Dropping a folder keeps its structure; the click-to-pick input is files
 * only (`webkitdirectory` would exclude files), matching its label.
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
  const { busy, outcomes, summary, handleFiles, handleDrop } = useUpload(
    config,
    onUploaded,
  );

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
          void handleDrop(event.dataTransfer);
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
          {busy ? "Uploading…" : "Drop files or folders here, or click to upload"}
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

      {summary !== null && (
        <p
          aria-live="polite"
          className="mt-2 rounded-md border border-border/60 px-2.5 py-1.5 text-xs text-muted-foreground"
        >
          {summary}
        </p>
      )}

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
