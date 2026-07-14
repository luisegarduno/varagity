/**
 * Fold the ingest-status SSE events into one renderable view.
 *
 * The stream replays a run from its first frame (spec_v2 §4.2), so the
 * reducer rebuilds identical state whether the client watched live,
 * connected mid-run, or connected after completion — the progress panel
 * has no other state source.
 */
import type { IngestEvent, IngestRun } from "@/lib/api";

/** One relayed pipeline log line. */
export interface IngestLogLine {
  level: string;
  message: string;
}

/** One file's final outcome (`file_done` frames). */
export interface FileOutcome {
  file: string;
  outcome: string;
  chunks: number | null;
}

export interface IngestView {
  /** The run snapshot (`null` until the first status frame / when idle). */
  run: IngestRun | null;
  /** Files discovered (`null` until discovery lands). */
  filesTotal: number | null;
  /** Files finished (any outcome). */
  filesDone: number;
  /** The file currently in the pipeline. */
  currentFile: string | null;
  /** Its current stage (`parse` | `chunk` | `contextualize` | `embed` | `store`). */
  currentStage: string | null;
  /** Per-chunk contextualization progress (the ingest's long pole). */
  contextualize: { done: number; total: number } | null;
  /** Relayed log tail, oldest first (capped). */
  logs: IngestLogLine[];
  /** Per-file outcomes, in completion order. */
  outcomes: FileOutcome[];
}

export const initialIngestView: IngestView = {
  run: null,
  filesTotal: null,
  filesDone: 0,
  currentFile: null,
  currentStage: null,
  contextualize: null,
  logs: [],
  outcomes: [],
};

/** Keep the log tail bounded on huge corpora. */
export const MAX_LOG_LINES = 500;

/** Apply one SSE event to the view (pure — returns a new object). */
export function reduceIngestEvent(view: IngestView, event: IngestEvent): IngestView {
  switch (event.type) {
    case "status": {
      const run = event.data.run ?? null;
      // A fresh "running" snapshot begins a new run: drop any previous
      // run's residue (the feed replays each run from scratch).
      const isNewRun =
        run !== null && run.state === "running" && run.run_id !== view.run?.run_id;
      const base = isNewRun ? initialIngestView : view;
      const terminal = run !== null && run.state !== "running";
      return {
        ...base,
        run,
        currentFile: terminal ? null : base.currentFile,
        currentStage: terminal ? null : base.currentStage,
        contextualize: terminal ? null : base.contextualize,
      };
    }
    case "progress": {
      const data = event.data;
      switch (data.stage) {
        case "discover":
          return { ...view, filesTotal: data.total ?? null };
        case "parse":
          return {
            ...view,
            currentFile: data.file ?? null,
            currentStage: "parse",
            contextualize: null,
          };
        case "contextualize":
          return {
            ...view,
            currentFile: data.file ?? view.currentFile,
            currentStage: "contextualize",
            contextualize:
              data.total != null ? { done: data.current ?? 0, total: data.total } : null,
          };
        case "chunk":
        case "embed":
        case "store":
          return {
            ...view,
            currentFile: data.file ?? view.currentFile,
            currentStage: data.stage,
            contextualize: null,
          };
        case "file_done":
          return {
            ...view,
            filesDone: data.files_done ?? view.filesDone + 1,
            filesTotal: data.files_total ?? view.filesTotal,
            currentFile: null,
            currentStage: null,
            contextualize: null,
            outcomes: [
              ...view.outcomes,
              {
                file: data.file ?? "(unknown)",
                outcome: data.outcome ?? "unknown",
                chunks: data.total ?? null,
              },
            ],
          };
        default:
          return view; // forward-compatible: unknown stages are ignored
      }
    }
    case "log": {
      const logs = [...view.logs, { level: event.data.level, message: event.data.message }];
      return { ...view, logs: logs.slice(-MAX_LOG_LINES) };
    }
    default:
      return view;
  }
}
