"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { DatabaseIcon, PowerOffIcon, Trash2Icon, UploadIcon } from "lucide-react";
import { useCallback, useRef, useState } from "react";

import { GraphBuildPanel } from "@/components/corpus/GraphBuildPanel";
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
import { useMountEffect } from "@/hooks/use-mount-effect";
import {
  ApiError,
  deleteGraphDocument,
  startGraphBuild,
  streamGraphBuildStatus,
  uploadGraphDocuments,
  type GraphDocument,
  type GraphStatus,
} from "@/lib/api";
import {
  initialGraphBuildView,
  reduceGraphBuildEvent,
  type GraphBuildView,
} from "@/lib/graph-build-reducer";
import { graphDocumentsQuery, graphStatusQuery, queryKeys } from "@/lib/queries";
import { REJECTION_LABELS } from "@/lib/use-upload";
import { cn } from "@/lib/utils";

const UNREACHABLE = "API unreachable — is the stack up? (docker compose up -d)";

/** One rendered upload outcome row. */
interface GraphUploadOutcome {
  fileName: string;
  ok: boolean;
  detail: string;
}

/** `12345678 → "11.8 MB"` — sizes here run from kilobytes to gigabytes. */
function formatBytes(bytes: number): string {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${unit === 0 ? value : value.toFixed(1)} ${units[unit]}`;
}

/** `2013-08-02T…` → `2013` — the parse summary's coverage is a year range. */
function year(iso: string | null | undefined): string | null {
  if (!iso) return null;
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? null : String(date.getFullYear());
}

/** The one-line "what's in this file" summary, or `null` before a scan. */
function parseLabel(document: GraphDocument): string | null {
  const parse = document.parse;
  if (!parse) return null;
  const from = year(parse.first);
  const to = year(parse.last);
  const span = from === null ? "" : from === to ? ` · ${from}` : ` · ${from}–${to}`;
  return `${parse.messages.toLocaleString()} messages · ${parse.threads} threads${span}`;
}

/**
 * The Graph RAG corpus tab (spec_graphrag §4.4): drop message archives in,
 * watch a resumable extraction build, and see what the graph holds.
 *
 * Deliberately unlike the document tab in three ways the subject matter
 * forces. Uploads are **sniffed, not filtered** — a copied `chat.db` is
 * routinely renamed, so the server decides by reading the stored bytes.
 * The listing is a **directory scan**, so the sidecars and contacts files
 * a user drops in are visible even though nothing parses them. And
 * deleting a source **flags the graph stale** rather than retracting its
 * messages: only a rebuild can do that, and saying so beats diverging
 * quietly (stage-2 decision #16).
 */
export function GraphCorpusView() {
  const queryClient = useQueryClient();
  const { data: status = null, error: statusError } = useQuery(
    graphStatusQuery(),
  );
  const { data: documents = null } = useQuery(graphDocumentsQuery());

  const [build, setBuild] = useState<GraphBuildView>(initialGraphBuildView);
  const [buildError, setBuildError] = useState<string | null>(null);
  const [uploads, setUploads] = useState<GraphUploadOutcome[]>([]);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [pending, setPending] = useState<GraphDocument | null>(null);
  const [deleting, setDeleting] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const followingRef = useRef(false);

  const enabled = status?.enabled ?? true;
  const stale = status?.stale ?? false;
  const running = build.run?.state === "running" || (status?.building ?? false);

  const statusErrorMessage =
    statusError === null
      ? null
      : statusError instanceof ApiError
        ? statusError.message
        : UNREACHABLE;
  const error = statusErrorMessage ?? buildError;

  const refresh = useCallback(() => {
    void queryClient.invalidateQueries({ queryKey: queryKeys.graphStatus });
    void queryClient.invalidateQueries({ queryKey: queryKeys.graphDocuments });
  }, [queryClient]);

  // Follow the current (or last) build. The stream replays from the run's
  // first frame, so attaching once on mount renders the same picture
  // whether a build is six hours in, finished, or absent entirely.
  const followBuild = useCallback(() => {
    if (followingRef.current) return;
    followingRef.current = true;
    void (async () => {
      try {
        let view = initialGraphBuildView;
        setBuild(view);
        for await (const event of streamGraphBuildStatus()) {
          view = reduceGraphBuildEvent(view, event);
          setBuild(view);
        }
      } catch {
        // Stream dropped (API restart, network): the panel keeps the last
        // known state; starting a build or reloading reconnects.
      } finally {
        followingRef.current = false;
        refresh();
      }
    })();
  }, [refresh]);

  useMountEffect(() => {
    followBuild();
  });

  const handleBuild = useCallback(
    async (options: { reingest?: boolean; messageLimit?: number | null }) => {
      try {
        await startGraphBuild({
          reingest: options.reingest ?? false,
          message_limit: options.messageLimit ?? null,
        });
        setBuildError(null);
        followBuild();
      } catch (failure) {
        setBuildError(
          failure instanceof ApiError
            ? failure.message
            : "Could not start the build — is the stack up?",
        );
      }
    },
    [followBuild],
  );

  const handleUpload = useCallback(
    async (files: File[]) => {
      if (files.length === 0 || uploading) return;
      setUploading(true);
      try {
        const response = await uploadGraphDocuments(files);
        setUploads(
          response.files.map((entry) => ({
            fileName: entry.file_name,
            ok: entry.stored,
            detail: entry.stored
              ? entry.replaced
                ? "replaced — re-build to pick up the new content"
                : "uploaded — not yet indexed"
              : ((entry.reason && REJECTION_LABELS[entry.reason]) ??
                entry.reason ??
                "rejected"),
          })),
        );
        refresh();
      } catch (failure) {
        setUploads([
          {
            fileName: `${files.length} file(s)`,
            ok: false,
            detail:
              failure instanceof ApiError ? failure.message : String(failure),
          },
        ]);
      } finally {
        setUploading(false);
      }
    },
    [refresh, uploading],
  );

  const confirmDelete = useCallback(async () => {
    if (pending === null || deleting) return;
    setDeleting(true);
    try {
      await deleteGraphDocument(pending.relative_path);
      setPending(null);
      refresh();
    } catch (failure) {
      setBuildError(
        failure instanceof ApiError ? failure.message : String(failure),
      );
    } finally {
      setDeleting(false);
    }
  }, [deleting, pending, refresh]);

  return (
    <div className="mx-auto flex w-full max-w-4xl flex-col gap-6 p-4 sm:p-6">
      {error && (
        <p
          role="alert"
          className="rounded-lg border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm text-destructive"
        >
          {error}
        </p>
      )}

      {!enabled && (
        <section
          aria-label="Graph disabled"
          className="flex flex-col items-center gap-2 rounded-xl border border-dashed border-border p-8 text-center"
        >
          <span
            aria-hidden
            className="flex size-9 items-center justify-center rounded-full bg-muted"
          >
            <PowerOffIcon className="size-4 text-muted-foreground" />
          </span>
          <p className="text-sm font-medium">The graph subsystem is off</p>
          <p className="max-w-prose text-xs text-muted-foreground">
            <code className="font-mono">GRAPH_ENABLED</code> is false, so
            uploads and builds are refused rather than silently ignored, and
            graph-targeted questions fall back to the document corpus. Turn it
            on in Settings to manage this corpus. Anything already extracted
            stays on disk.
          </p>
        </section>
      )}

      {stale && (
        <div
          role="status"
          className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm"
        >
          <span className="flex min-w-0 items-center gap-2.5">
            <Badge variant="warning">stale</Badge>
            <span>
              A source file was removed since the last build — the graph still
              holds its messages until it is rebuilt.
            </span>
          </span>
          <Button
            variant="outline"
            size="sm"
            disabled={running || !enabled}
            onClick={() => void handleBuild({ reingest: true })}
          >
            Re-build to apply
          </Button>
        </div>
      )}

      <GraphStats status={status} />

      {enabled && (
        <section aria-label="Upload message archives">
          <div
            role="button"
            tabIndex={0}
            aria-label="Upload message archives into the graph corpus"
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
              if (
                !event.currentTarget.contains(event.relatedTarget as Node | null)
              ) {
                setDragging(false);
              }
            }}
            onDrop={(event) => {
              event.preventDefault();
              setDragging(false);
              void handleUpload(Array.from(event.dataTransfer.files));
            }}
            className={cn(
              "flex cursor-pointer flex-col items-center gap-2 rounded-xl border border-dashed border-border p-8 text-center select-none motion-safe:transition-colors motion-safe:duration-150",
              dragging
                ? "border-ring bg-primary/5"
                : "hover:border-muted-foreground/40 hover:bg-muted/30",
              uploading && "pointer-events-none opacity-60",
            )}
          >
            <span
              aria-hidden
              className="flex size-9 items-center justify-center rounded-full bg-muted"
            >
              <UploadIcon className="size-4 text-muted-foreground" />
            </span>
            <p className="text-sm font-medium">
              {uploading
                ? "Uploading…"
                : "Drop a message archive here, or click to upload"}
            </p>
            <p className="max-w-prose text-xs text-muted-foreground">
              An iMessage <code className="font-mono">chat.db</code> and its
              two sidecars (<code className="font-mono">-wal</code>,{" "}
              <code className="font-mono">-shm</code>) — copy all three, or the
              most recent messages are silently missing. The server decides
              what a file is by reading it, not by its name.
            </p>
            <input
              ref={inputRef}
              type="file"
              multiple
              className="hidden"
              onChange={(event) => {
                void handleUpload(Array.from(event.target.files ?? []));
                event.target.value = ""; // allow re-picking the same file
              }}
            />
          </div>

          {uploads.length > 0 && (
            <ul className="mt-2 flex flex-col gap-1" aria-label="Upload outcomes">
              {uploads.map((outcome, index) => (
                <li
                  key={`${outcome.fileName}-${index}`}
                  className="flex flex-wrap items-center gap-2 rounded-md border border-border/60 px-2.5 py-1.5 text-xs"
                >
                  <Badge variant={outcome.ok ? "success" : "destructive"}>
                    {outcome.ok ? "uploaded" : "rejected"}
                  </Badge>
                  <span
                    className="max-w-56 truncate font-medium"
                    title={outcome.fileName}
                  >
                    {outcome.fileName}
                  </span>
                  <span className="text-muted-foreground">{outcome.detail}</span>
                </li>
              ))}
            </ul>
          )}
        </section>
      )}

      {enabled && (
        <GraphBuildPanel
          view={build}
          disabled={running}
          onBuild={({ messageLimit }) => void handleBuild({ messageLimit })}
          onRebuild={({ messageLimit }) =>
            void handleBuild({ reingest: true, messageLimit })
          }
        />
      )}

      <section aria-label="Graph sources" className="flex flex-col gap-2">
        <h2 className="text-sm font-semibold">Sources</h2>
        {documents === null ? (
          <div className="overflow-hidden rounded-lg border border-border">
            {[0, 1].map((row) => (
              <div
                key={row}
                className="flex items-center gap-4 border-t border-border px-3 py-3 first:border-t-0"
              >
                <Skeleton className="h-4 w-44" />
                <Skeleton className="ml-auto h-4 w-28" />
              </div>
            ))}
          </div>
        ) : documents.length === 0 ? (
          <p className="text-xs text-muted-foreground">
            Nothing here yet — drop a message archive above, then build.
          </p>
        ) : (
          <div className="overflow-x-auto rounded-lg border border-border">
            <table className="w-full text-left text-sm">
              <thead className="bg-muted/40 text-xs text-muted-foreground">
                <tr>
                  <th className="px-3 py-2 font-medium">File</th>
                  <th className="px-3 py-2 font-medium">Size</th>
                  <th className="px-3 py-2 font-medium">Parsed</th>
                  <th className="px-3 py-2">
                    <span className="sr-only">Actions</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {documents.map((document) => {
                  const parsed = parseLabel(document);
                  return (
                    <tr
                      key={document.relative_path}
                      className="border-t border-border transition-colors hover:bg-muted/40"
                    >
                      <td
                        className="max-w-64 px-3 py-2"
                        title={document.relative_path}
                      >
                        <span className="flex min-w-0 items-center gap-1.5">
                          <DatabaseIcon
                            aria-hidden
                            className="size-3.5 shrink-0 text-muted-foreground"
                          />
                          <span className="truncate">
                            {document.relative_path}
                          </span>
                        </span>
                      </td>
                      <td className="px-3 py-2 font-mono text-xs tabular-nums">
                        {formatBytes(document.size_bytes)}
                      </td>
                      <td className="px-3 py-2 text-xs text-muted-foreground">
                        {parsed ?? (
                          <span title="No build has scanned this file — parsing a multi-gigabyte archive is a build's job">
                            not scanned
                          </span>
                        )}
                      </td>
                      <td className="px-3 py-2 text-right">
                        <Button
                          variant="ghost"
                          size="icon-sm"
                          aria-label={`Delete ${document.relative_path}`}
                          onClick={() => setPending(document)}
                        >
                          <Trash2Icon className="size-4" />
                        </Button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <AlertDialog
        open={pending !== null}
        onOpenChange={(open) => {
          if (!open) setPending(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              Delete “{pending?.relative_path}”?
            </AlertDialogTitle>
            <AlertDialogDescription>
              Removes the file from the graph corpus directory. The graph keeps
              the messages it already extracted from it — only a re-build
              retracts them — so the graph is flagged stale until you run one.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogClose render={<Button variant="outline" />}>
              Cancel
            </AlertDialogClose>
            <Button
              variant="destructive"
              onClick={() => void confirmDelete()}
              disabled={deleting}
            >
              {deleting ? "Deleting…" : "Delete"}
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

/**
 * What the graph holds, straight off the workdir's summary sidecar (never
 * a per-request graph walk). Every figure is optional on the wire because
 * "nothing built here yet" and "the engine would not say" are both real
 * states, and a `0` would read like an empty graph.
 */
function GraphStats({ status }: { status: GraphStatus | null }) {
  const figures: [string, number | null][] = [
    ["entities", status?.entities ?? null],
    ["relations", status?.relations ?? null],
    ["messages", status?.message_guids ?? null],
  ];
  const statuses = Object.entries(status?.documents ?? {});
  return (
    <dl
      aria-label="Graph size"
      className="flex flex-wrap items-baseline gap-x-6 gap-y-2 rounded-lg border border-border px-4 py-3"
    >
      {figures.map(([label, value]) => (
        <div key={label} className="flex flex-col">
          <dt className="text-xs text-muted-foreground">{label}</dt>
          <dd className="font-mono text-lg tabular-nums">
            {value === null ? "—" : value.toLocaleString()}
          </dd>
        </div>
      ))}
      <div className="flex min-w-0 flex-col">
        <dt className="text-xs text-muted-foreground">thread-days</dt>
        <dd className="flex flex-wrap items-center gap-1.5">
          {statuses.length === 0 ? (
            <span className="font-mono text-lg tabular-nums">—</span>
          ) : (
            statuses.map(([name, count]) => (
              <Badge
                key={name}
                variant={name === "failed" && count > 0 ? "destructive" : "default"}
                className="font-mono"
              >
                {name} {count}
              </Badge>
            ))
          )}
        </dd>
      </div>
    </dl>
  );
}
