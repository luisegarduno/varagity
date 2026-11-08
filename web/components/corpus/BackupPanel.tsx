"use client";

import {
  ArchiveIcon,
  CheckIcon,
  CopyIcon,
  DownloadIcon,
  TriangleAlertIcon,
} from "lucide-react";
import { useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button, buttonVariants } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { backupFileUrl, type BundleOut, type TargetCorpus } from "@/lib/api";
import {
  backupStageLabel,
  backupStageUnit,
  describeBundle,
  formatBytes,
  type BackupView,
} from "@/lib/backup-reducer";
import { cn } from "@/lib/utils";

/** Per-corpus copy: where its bundles live and what a bundle saves. */
const COPY: Record<TargetCorpus, { folder: string; blurb: string; empty: string }> = {
  rag: {
    folder: "docs/varagity-exports/",
    blurb:
      "Save the processed corpus — chunks, contextual blurbs and embeddings — to one file. It lands beside your documents, so it survives docker compose down -v.",
    empty: "nothing ingested yet",
  },
  graph: {
    folder: "graph-docs/varagity-exports/",
    blurb:
      "Save the extracted graph — entities, relations, vectors, the LLM cache and the build manifest — to one file.",
    empty: "nothing built yet",
  },
};

const STATE_BADGE_VARIANT: Record<string, "accent" | "success" | "destructive"> = {
  running: "accent",
  completed: "success",
  failed: "destructive",
};

/** The bundle's own instant when its manifest says, else the file's mtime. */
function bundleTime(bundle: BundleOut): string {
  const when = new Date(bundle.manifest?.created_at ?? bundle.modified_at);
  return when.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

/** Copy-path button with its own brief "Copied" / "Copy failed" feedback. */
function CopyPathButton({ path }: { path: string }) {
  const [outcome, setOutcome] = useState<"idle" | "copied" | "failed">("idle");

  async function copy() {
    try {
      // The Clipboard API exists only in secure contexts (https, localhost);
      // over a LAN http origin the path stays selectable instead.
      if (!navigator.clipboard) throw new Error("clipboard unavailable");
      await navigator.clipboard.writeText(path);
      setOutcome("copied");
    } catch {
      setOutcome("failed");
    }
    setTimeout(() => setOutcome("idle"), 1500);
  }

  return (
    <Button variant="outline" size="xs" onClick={() => void copy()}>
      {outcome === "copied" ? <CheckIcon aria-hidden /> : <CopyIcon aria-hidden />}
      {outcome === "copied" ? "Copied" : outcome === "failed" ? "Copy failed" : "Copy path"}
    </Button>
  );
}

/**
 * The "Export & restore" panel (ADR-019) — presentational, like
 * `IngestPanel`: the corpus tab's `useBackupRun` owns the stream and the
 * bundle list, and this renders them.
 *
 * **Export** writes one bundle into the corpus folder's reserved
 * `varagity-exports/` and the panel then shows exactly where it went, with
 * Copy path and Download — no browser can open a folder on the user's
 * machine, so the saved path *is* the "open the folder" answer. Download
 * is a plain attachment link (no fetch, no CORS). Below that, every bundle
 * in the folder, newest first, so a backup copied in by hand shows up too.
 */
export function BackupPanel({
  corpus,
  view,
  bundles,
  bundlesError = null,
  holding,
  empty,
  error,
  onExport,
}: {
  corpus: TargetCorpus;
  view: BackupView;
  /** The corpus's bundles, newest first (`null` while loading). */
  bundles: BundleOut[] | null;
  bundlesError?: string | null;
  /** What the corpus holds now (`"142 documents · 10,233 chunks"`); `null` when unknown. */
  holding: string | null;
  /** The corpus is confirmed empty — there is nothing to export. */
  empty: boolean;
  /** Why the last Export was refused. */
  error: string | null;
  onExport: () => void;
}) {
  const copy = COPY[corpus];
  const run = view.run;
  const running = run?.state === "running";
  const stage = backupStageLabel(view);
  const fresh = run?.state === "completed" ? run.bundle : null;

  return (
    <section
      aria-label="Export and restore"
      className="flex flex-col gap-3 rounded-lg border border-border p-4"
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0 flex-1 basis-64">
          <h2 className="text-sm font-semibold">Export &amp; restore</h2>
          <p className="text-xs text-muted-foreground">{copy.blurb}</p>
        </div>
        <Button size="sm" onClick={onExport} disabled={empty || running}>
          <ArchiveIcon aria-hidden /> Export
        </Button>
      </div>

      {(holding !== null || empty) && (
        <p className="text-xs text-muted-foreground">
          Now: {holding ?? copy.empty} · saves to{" "}
          <code className="font-mono break-all">{copy.folder}</code>
        </p>
      )}

      {error && (
        <p role="alert" className="text-xs text-destructive">
          {error}
        </p>
      )}

      {run === null ? (
        <p className="text-xs text-muted-foreground">
          No export has run since the API started.
        </p>
      ) : (
        <div className="flex flex-col gap-2">
          {/* role=status: run-state flips are announced politely; the
              per-stage churn below stays quiet. */}
          <div role="status" className="flex flex-wrap items-center gap-2 text-xs">
            <Badge variant={STATE_BADGE_VARIANT[run.state] ?? "default"}>
              {running ? <span className="shimmer">running</span> : run.state}
            </Badge>
            <span className="font-mono text-muted-foreground">
              run {run.run_id} · {run.operation}
            </span>
            {running && stage && <span className="text-muted-foreground">{stage}</span>}
          </div>

          {running && view.total !== null && view.total > 0 && (
            <div>
              <div className="mb-1.5 text-xs text-muted-foreground tabular-nums">
                {(view.current ?? 0).toLocaleString()} / {view.total.toLocaleString()}{" "}
                {backupStageUnit(view.stage)}
              </div>
              <div
                role="progressbar"
                aria-label="Export progress"
                aria-valuemin={0}
                aria-valuemax={view.total}
                aria-valuenow={view.current ?? 0}
                aria-valuetext={`${view.current ?? 0} of ${view.total} ${backupStageUnit(view.stage)}`}
                className="h-1.5 overflow-hidden rounded-full bg-muted"
              >
                <div
                  className="h-full rounded-full bg-primary transition-[width] duration-300 motion-reduce:transition-none"
                  style={{
                    width: `${Math.min(100, ((view.current ?? 0) / view.total) * 100)}%`,
                  }}
                />
              </div>
            </div>
          )}

          {run.state === "completed" && run.saved_to && run.bundle && (
            <div className="flex flex-wrap items-center gap-2 rounded-md bg-muted/65 px-3 py-2">
              <span className="min-w-0 flex-1 basis-64 text-xs">
                Saved to{" "}
                <code className="font-mono break-all select-all">{run.saved_to}</code>
                {run.summary && (
                  <span className="text-muted-foreground">
                    {" "}
                    · {formatBytes(run.summary.size_bytes)}
                  </span>
                )}
              </span>
              <span className="flex flex-wrap gap-2">
                <CopyPathButton path={run.saved_to} />
                <a
                  href={backupFileUrl(corpus, run.bundle)}
                  download={run.bundle}
                  className={cn(buttonVariants({ variant: "outline", size: "xs" }))}
                >
                  <DownloadIcon aria-hidden /> Download
                </a>
              </span>
            </div>
          )}

          {run.state === "failed" && run.error && (
            <p role="alert" className="text-xs text-destructive">
              {run.error}
            </p>
          )}

          {view.logs.length > 0 && (
            <details className="text-xs" open={running}>
              <summary className="cursor-pointer text-muted-foreground transition-colors select-none hover:text-foreground">
                Export log ({view.logs.length})
              </summary>
              <div className="mt-1.5 max-h-48 overflow-y-auto rounded-md border border-border bg-muted/30 p-2 font-mono text-xs leading-relaxed text-muted-foreground scroll-fade-y">
                {view.logs.map((line, index) => (
                  <p
                    key={index}
                    className={cn(
                      line.level === "ERROR" && "text-destructive",
                      line.level === "WARNING" && "text-amber-600 dark:text-amber-400",
                    )}
                  >
                    {line.message}
                  </p>
                ))}
              </div>
            </details>
          )}
        </div>
      )}

      <div className="flex flex-col gap-1.5">
        <p className="text-[11px] font-medium tracking-[0.08em] text-muted-foreground uppercase">
          Backups in {copy.folder}
        </p>
        {bundlesError !== null ? (
          <p className="text-xs text-destructive">{bundlesError}</p>
        ) : bundles === null ? (
          <div className="flex flex-col gap-2" aria-hidden>
            <Skeleton className="h-4 w-72" />
            <Skeleton className="h-4 w-48" />
          </div>
        ) : bundles.length === 0 ? (
          <p className="text-xs text-muted-foreground">
            No backups yet — Export writes the first one here.
          </p>
        ) : (
          <ul
            aria-label="Backups"
            className="divide-y divide-border rounded-md border border-border"
          >
            {bundles.map((bundle) => {
              const description = bundle.manifest ? describeBundle(bundle.manifest) : null;
              return (
                <li
                  key={bundle.name}
                  className="flex flex-wrap items-center justify-between gap-x-3 gap-y-2 px-3 py-2.5"
                >
                  <div className="flex min-w-0 flex-1 basis-72 flex-col gap-0.5">
                    <span className="flex flex-wrap items-center gap-1.5 text-xs">
                      <span className="font-mono break-all">{bundle.name}</span>
                      {bundle.name === fresh && <Badge variant="accent">new</Badge>}
                    </span>
                    <span className="text-xs text-muted-foreground">
                      {bundleTime(bundle)} · {formatBytes(bundle.size_bytes)}
                    </span>
                    {description ? (
                      <span className="text-xs">
                        {description.contents}
                        {description.settings && (
                          <span className="text-muted-foreground">
                            {" "}
                            · {description.settings}
                          </span>
                        )}
                      </span>
                    ) : (
                      <span className="flex items-start gap-1.5 text-xs text-amber-700 dark:text-amber-400">
                        <TriangleAlertIcon className="mt-0.5 size-3.5 shrink-0" aria-hidden />
                        No readable bundle.json — not a Varagity backup.
                      </span>
                    )}
                  </div>
                  <a
                    href={backupFileUrl(corpus, bundle.name)}
                    download={bundle.name}
                    aria-label={`Download ${bundle.name}`}
                    title="Download"
                    className={cn(buttonVariants({ variant: "ghost", size: "icon-xs" }))}
                  >
                    <DownloadIcon aria-hidden />
                  </a>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </section>
  );
}
