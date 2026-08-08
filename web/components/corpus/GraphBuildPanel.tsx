"use client";

import { HammerIcon, RefreshCwIcon } from "lucide-react";
import { useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  graphStageLabel,
  type GraphBuildView,
} from "@/lib/graph-build-reducer";
import { cn } from "@/lib/utils";

const STATE_BADGE_VARIANT: Record<string, "accent" | "success" | "destructive"> =
  {
    running: "accent",
    completed: "success",
    failed: "destructive",
  };

/** Whole-number percentage of a `done / total` pair, clamped to 100. */
function percent(done: number, total: number): number {
  return total <= 0 ? 0 : Math.min(100, (done / total) * 100);
}

/**
 * The graph build's controls and live progress (spec_graphrag §5.2) — the
 * `IngestPanel` shape over the build-status SSE.
 *
 * Two things make this panel different from the ingest one, and both are
 * the subject matter rather than the styling: a full backfill runs for
 * hours to days (so the copy says so, and the bounded escape hatch is one
 * disclosure away), and pressing **Build** on an interrupted corpus
 * *resumes* it — the engine keeps its document statuses on disk, so the
 * work already done is never repeated.
 */
export function GraphBuildPanel({
  view,
  disabled,
  onBuild,
  onRebuild,
}: {
  view: GraphBuildView;
  disabled: boolean;
  /** `message_limit` bounds the run to the newest N messages. */
  onBuild: (options: { messageLimit: number | null }) => void;
  /** Same bound — a typed limit applies to a from-scratch run too. */
  onRebuild: (options: { messageLimit: number | null }) => void;
}) {
  const [advanced, setAdvanced] = useState(false);
  const [limit, setLimit] = useState("");

  const run = view.run;
  const running = run?.state === "running";
  const summary = run?.summary ?? null;
  const stage = graphStageLabel(view);
  const parsedLimit = Number.parseInt(limit, 10);
  const messageLimit =
    Number.isFinite(parsedLimit) && parsedLimit > 0 ? parsedLimit : null;

  return (
    <section
      aria-label="Graph build"
      className="flex flex-col gap-3 rounded-lg border border-border p-4"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h2 className="text-sm font-semibold">Build</h2>
          <p className="text-xs text-muted-foreground">
            Parse the archives, then extract entities and relations one
            thread-day at a time.
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            size="sm"
            onClick={() => onBuild({ messageLimit })}
            disabled={disabled}
          >
            <HammerIcon aria-hidden /> Build
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => onRebuild({ messageLimit })}
            disabled={disabled}
          >
            <RefreshCwIcon aria-hidden /> Re-build from scratch
          </Button>
        </div>
      </div>

      <p className="text-xs text-muted-foreground">
        A full backfill of a decade-scale archive runs for hours to days. It
        can be stopped at any time — pressing <strong>Build</strong> again
        picks up exactly where it left off, and never re-extracts a
        thread-day it already finished. <strong>Re-build from scratch</strong>
        throws the graph away and starts over; it is the only thing that
        clears the stale flag.
      </p>

      <div>
        <Button
          type="button"
          variant="ghost"
          size="xs"
          className="-ml-1 text-muted-foreground"
          aria-expanded={advanced}
          onClick={() => setAdvanced((open) => !open)}
        >
          {advanced ? "Hide bounds" : "Bound this build"}
        </Button>
        {advanced && (
          <div className="mt-1.5 flex flex-wrap items-center gap-2">
            <label
              htmlFor="graph-message-limit"
              className="text-xs text-muted-foreground"
            >
              Newest messages only
            </label>
            <Input
              id="graph-message-limit"
              type="number"
              min={1}
              inputMode="numeric"
              placeholder="all"
              value={limit}
              onChange={(event) => setLimit(event.target.value)}
              className="h-8 w-28"
            />
            <span className="text-xs text-muted-foreground">
              a spot check in minutes instead of days (a bounded build never
              prunes)
            </span>
          </div>
        )}
      </div>

      {run === null ? (
        <p className="text-xs text-muted-foreground">
          No build has run since the API started.
        </p>
      ) : (
        <div className="flex flex-col gap-2">
          {/* role=status: run-state flips are announced politely; the
              per-document churn below stays quiet. */}
          <div role="status" className="flex flex-wrap items-center gap-2 text-xs">
            <Badge variant={STATE_BADGE_VARIANT[run.state] ?? "default"}>
              {running ? <span className="shimmer">running</span> : run.state}
            </Badge>
            <span className="font-mono text-muted-foreground">
              run {run.run_id}
              {run.reingest ? " · re-build" : ""}
              {run.bounded ? " · bounded" : ""}
            </span>
            {running && stage && (
              <span className="text-muted-foreground">
                {stage}
                {view.currentFile && (
                  <span className="font-mono text-foreground">
                    {" "}
                    {view.currentFile}
                  </span>
                )}
              </span>
            )}
          </div>

          {view.docsTotal !== null && view.docsTotal > 0 && (
            <div>
              <div className="mb-1.5 flex justify-between gap-3 text-xs text-muted-foreground">
                <span className="shrink-0 tabular-nums">
                  {view.docsDone ?? 0} / {view.docsTotal} thread-days
                </span>
                {view.messagesIndexed !== null && (
                  <span className="tabular-nums">
                    {view.messagesIndexed} messages indexed
                  </span>
                )}
              </div>
              <div
                role="progressbar"
                aria-label="Thread-day extraction progress"
                aria-valuemin={0}
                aria-valuemax={view.docsTotal}
                aria-valuenow={view.docsDone ?? 0}
                aria-valuetext={`${view.docsDone ?? 0} of ${view.docsTotal} thread-days`}
                className="h-1.5 overflow-hidden rounded-full bg-muted"
              >
                <div
                  className="h-full rounded-full bg-primary transition-[width] duration-300 motion-reduce:transition-none"
                  style={{
                    width: `${percent(view.docsDone ?? 0, view.docsTotal)}%`,
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
            <dl
              className="flex flex-wrap gap-x-4 gap-y-1 text-xs"
              aria-label="Build summary"
            >
              {(
                [
                  ["files parsed", summary.files_parsed],
                  ["messages indexed", summary.messages_indexed],
                  ["resumed", summary.resumed],
                  ["files failed", summary.files_failed],
                ] as const
              ).map(([label, count]) => (
                <div key={label} className="flex gap-1">
                  <dt className="text-muted-foreground">{label}</dt>
                  <dd
                    className={cn(
                      "font-medium tabular-nums",
                      label === "files failed" &&
                        count > 0 &&
                        "text-destructive",
                    )}
                  >
                    {count}
                  </dd>
                </div>
              ))}
              <div className="flex gap-1">
                <dt className="text-muted-foreground">wall clock</dt>
                <dd className="font-medium tabular-nums">
                  {(summary.wall_clock_s / 60).toFixed(1)} min
                </dd>
              </div>
            </dl>
          )}

          {summary && summary.failures.length > 0 && (
            <ul className="text-xs text-destructive" aria-label="Build failures">
              {summary.failures.map((failure) => (
                <li key={failure}>{failure}</li>
              ))}
            </ul>
          )}

          {view.logs.length > 0 && (
            <details className="text-xs" open={running}>
              <summary className="cursor-pointer text-muted-foreground transition-colors select-none hover:text-foreground">
                Build log ({view.logs.length})
              </summary>
              <div className="mt-1.5 max-h-48 overflow-y-auto rounded-md border border-border bg-muted/30 p-2 font-mono text-xs leading-relaxed text-muted-foreground scroll-fade-y">
                {view.logs.map((line, index) => (
                  <p
                    key={index}
                    className={cn(
                      line.level === "ERROR" && "text-destructive",
                      line.level === "WARNING" &&
                        "text-amber-600 dark:text-amber-400",
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
