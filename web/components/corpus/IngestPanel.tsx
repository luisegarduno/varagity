"use client";

import { PlayIcon, RefreshCwIcon } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { IngestView } from "@/lib/ingest-reducer";
import { cn } from "@/lib/utils";

const STAGE_LABELS: Record<string, string> = {
  parse: "parsing",
  chunk: "chunking",
  contextualize: "contextualizing",
  embed: "embedding",
  store: "storing",
};

const STATE_BADGE_VARIANT: Record<string, "accent" | "success" | "destructive"> = {
  running: "accent",
  completed: "success",
  failed: "destructive",
};

/**
 * The live ingest-progress view (spec_v2 §4.2), fed by the status SSE via
 * the reducer: run controls, a files bar, the current stage (with
 * per-chunk contextualize ticks — the long pole), the relayed log tail,
 * and the terminal summary counters.
 */
export function IngestPanel({
  view,
  disabled,
  onIngest,
  onReingest,
}: {
  view: IngestView;
  disabled: boolean;
  onIngest: () => void;
  onReingest: () => void;
}) {
  const run = view.run;
  const running = run?.state === "running";
  const summary = run?.summary ?? null;

  return (
    <section
      aria-label="Ingestion"
      className="flex flex-col gap-3 rounded-lg border border-border p-4"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h2 className="text-sm font-semibold">Ingestion</h2>
          <p className="text-xs text-muted-foreground">
            Parse → chunk → contextualize → embed → store, into both stores.
          </p>
        </div>
        <div className="flex gap-2">
          <Button size="sm" onClick={onIngest} disabled={disabled}>
            <PlayIcon aria-hidden /> Ingest new
          </Button>
          <Button size="sm" variant="outline" onClick={onReingest} disabled={disabled}>
            <RefreshCwIcon aria-hidden /> Re-ingest all
          </Button>
        </div>
      </div>

      {run === null ? (
        <p className="text-xs text-muted-foreground">
          No ingest has run since the API started.
        </p>
      ) : (
        <div className="flex flex-col gap-2">
          {/* role=status: run-state flips (running → completed/failed) are
              announced politely; the per-file churn below stays quiet. */}
          <div role="status" className="flex items-center gap-2 text-xs">
            <Badge variant={STATE_BADGE_VARIANT[run.state] ?? "default"}>
              {running ? <span className="shimmer">running</span> : run.state}
            </Badge>
            <span className="font-mono text-muted-foreground">
              run {run.run_id}
              {run.reingest ? " · re-ingest" : ""}
            </span>
          </div>

          {view.filesTotal !== null && view.filesTotal > 0 && (
            <div>
              <div className="mb-1.5 flex justify-between gap-3 text-xs text-muted-foreground">
                <span className="shrink-0 tabular-nums">
                  {view.filesDone} / {view.filesTotal} files
                </span>
                {running && view.currentFile && (
                  <span className="min-w-0 truncate font-mono">
                    {STAGE_LABELS[view.currentStage ?? ""] ?? view.currentStage}{" "}
                    <span className="text-foreground">{view.currentFile}</span>
                    {view.contextualize && (
                      <span className="tabular-nums">
                        {" "}
                        ({view.contextualize.done}/{view.contextualize.total} chunks)
                      </span>
                    )}
                  </span>
                )}
              </div>
              <div
                role="progressbar"
                aria-label="Files progress"
                aria-valuemin={0}
                aria-valuemax={view.filesTotal}
                aria-valuenow={view.filesDone}
                aria-valuetext={`${view.filesDone} of ${view.filesTotal} files`}
                className="h-1.5 overflow-hidden rounded-full bg-muted"
              >
                <div
                  className="h-full rounded-full bg-primary transition-[width] duration-300 motion-reduce:transition-none"
                  style={{
                    width: `${Math.min(100, (view.filesDone / view.filesTotal) * 100)}%`,
                  }}
                />
              </div>
            </div>
          )}

          {run.state === "failed" && run.error && (
            <p role="alert" className="text-xs text-destructive">
              {run.error}
            </p>
          )}

          {summary && (
            <dl className="flex flex-wrap gap-x-4 gap-y-1 text-xs" aria-label="Run summary">
              {(
                [
                  ["discovered", summary.discovered],
                  ["ingested", summary.ingested],
                  ["skipped", summary.skipped],
                  ["no text", summary.no_text],
                  ["failed", summary.failed],
                  ["chunks", summary.chunks],
                ] as const
              ).map(([label, count]) => (
                <div key={label} className="flex gap-1">
                  <dt className="text-muted-foreground">{label}</dt>
                  <dd
                    className={cn(
                      "font-medium tabular-nums",
                      label === "failed" && count > 0 && "text-destructive",
                    )}
                  >
                    {count}
                  </dd>
                </div>
              ))}
            </dl>
          )}

          {view.logs.length > 0 && (
            <details className="text-xs" open={running}>
              <summary className="cursor-pointer text-muted-foreground transition-colors select-none hover:text-foreground">
                Pipeline log ({view.logs.length})
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
    </section>
  );
}
