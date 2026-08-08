/**
 * Fold the graph-build SSE events into one renderable view.
 *
 * The `lib/ingest-reducer.ts` twin (spec_graphrag §5.2), and pure for the
 * same reason: the stream replays a run from its first frame, so the
 * reducer must rebuild identical state whether the client watched live,
 * connected six hours into a backfill, or connected after it finished —
 * the build panel has no other state source.
 *
 * A graph build runs on two clocks, and the view carries both: the corpus
 * walk (`scan` → `parse` per file → `bound`/`reset`) takes seconds and is
 * counted in *files*, while the extraction pass (`index` → sampled
 * `process` ticks) runs for hours and is counted in the engine's own
 * *documents* (thread-days). Only the second one is worth a progress bar.
 */
import type { GraphBuildEvent, GraphBuildRun } from "@/lib/api";

/** One relayed `varagity.graph` log line. */
export interface GraphBuildLogLine {
  level: string;
  message: string;
}

export interface GraphBuildView {
  /** The run snapshot (`null` until the first status frame / when idle). */
  run: GraphBuildRun | null;
  /**
   * The stage the run last reported: `scan` | `parse` | `bound` | `reset` |
   * `index` | `process`. `null` before the first frame and once terminal.
   */
  stage: string | null;
  /** Files found under `GRAPH_DOCS_PATH` (`null` until the scan lands). */
  filesTotal: number | null;
  /** Files the scan has walked so far. */
  filesDone: number;
  /** The file being parsed (`null` outside `parse`). */
  currentFile: string | null;
  /** Messages parsed out of the corpus (`null` until `bound`/`index`). */
  messagesParsed: number | null;
  /** Messages handed to the engine after the bounds applied. */
  messagesIndexed: number | null;
  /** Documents the engine reports processed (the extraction pass's clock). */
  docsDone: number | null;
  /** Documents the engine holds in any status. */
  docsTotal: number | null;
  /** Relayed log tail, oldest first (capped). */
  logs: GraphBuildLogLine[];
}

export const initialGraphBuildView: GraphBuildView = {
  run: null,
  stage: null,
  filesTotal: null,
  filesDone: 0,
  currentFile: null,
  messagesParsed: null,
  messagesIndexed: null,
  docsDone: null,
  docsTotal: null,
  logs: [],
};

/** Keep the log tail bounded on a multi-day backfill. */
export const MAX_GRAPH_LOG_LINES = 500;

/** Apply one SSE event to the view (pure — returns a new object). */
export function reduceGraphBuildEvent(
  view: GraphBuildView,
  event: GraphBuildEvent,
): GraphBuildView {
  switch (event.type) {
    case "status": {
      const run = event.data.run ?? null;
      // A fresh "running" snapshot begins a new run: drop the previous
      // run's residue (the feed replays each run from scratch).
      const isNewRun =
        run !== null &&
        run.state === "running" &&
        run.run_id !== view.run?.run_id;
      const base = isNewRun ? initialGraphBuildView : view;
      const terminal = run !== null && run.state !== "running";
      // A completed run's summary is the authoritative document tally —
      // the sampled `process` ticks can freeze one short of the end.
      const documents =
        run?.state === "completed" ? (run.summary?.documents ?? null) : null;
      return {
        ...base,
        run,
        stage: terminal ? null : base.stage,
        currentFile: terminal ? null : base.currentFile,
        docsDone: documents ? (documents.processed ?? 0) : base.docsDone,
        docsTotal: documents
          ? Object.values(documents).reduce((sum, count) => sum + count, 0)
          : base.docsTotal,
      };
    }
    case "progress": {
      const data = event.data;
      switch (data.stage) {
        case "scan":
          return {
            ...view,
            stage: "scan",
            filesTotal: data.total ?? null,
            filesDone: 0,
            currentFile: null,
          };
        case "parse":
          return {
            ...view,
            stage: "parse",
            currentFile: data.file ?? null,
            filesDone: data.current ?? view.filesDone,
            filesTotal: data.total ?? view.filesTotal,
          };
        case "bound":
          // The bounded escape hatch: `current` kept of `total` parsed.
          return {
            ...view,
            stage: "bound",
            currentFile: null,
            messagesIndexed: data.current ?? view.messagesIndexed,
            messagesParsed: data.total ?? view.messagesParsed,
          };
        case "reset":
          return { ...view, stage: "reset", currentFile: null };
        case "index":
          // Handing off to the engine: the file clock is done, and `total`
          // is what an unbounded run never announced at `bound`.
          return {
            ...view,
            stage: "index",
            currentFile: null,
            messagesIndexed: data.total ?? view.messagesIndexed,
          };
        case "process":
          return {
            ...view,
            stage: "process",
            currentFile: null,
            docsDone: data.docs_done ?? view.docsDone,
            docsTotal: data.docs_total ?? view.docsTotal,
          };
        default:
          return view; // forward-compatible: unknown stages are ignored
      }
    }
    case "log": {
      const logs = [
        ...view.logs,
        { level: event.data.level, message: event.data.message },
      ];
      return { ...view, logs: logs.slice(-MAX_GRAPH_LOG_LINES) };
    }
    default:
      return view;
  }
}

/** Human label for the stage the build is in (`null` when idle). */
export function graphStageLabel(view: GraphBuildView): string | null {
  switch (view.stage) {
    case "scan":
      return "scanning the corpus";
    case "parse":
      return "parsing";
    case "bound":
      return "applying the message bound";
    case "reset":
      return "wiping the working directory";
    case "index":
      return "handing messages to the engine";
    case "process":
      return "extracting entities and relations";
    default:
      return null;
  }
}
