/**
 * Fold the backup-status SSE events into one renderable view (ADR-019).
 *
 * The `lib/ingest-reducer.ts` twin, and pure for the same reason: the stream
 * replays a run from its first frame, so the reducer must rebuild identical
 * state whether the panel watched the export live, connected mid-run, or
 * connected after it finished — the backup panel has no other state source.
 * Both corpora's panels fold their own stream with it.
 *
 * Also home to the panel's small presentation helpers (stage labels, bundle
 * summaries, byte sizes), which are pure and so live here, under the
 * Vitest suite, rather than in the component.
 */
import type {
  BackupEvent,
  BackupRun,
  BackupSummary,
  BundleManifest,
  BundleOut,
} from "@/lib/api";

/** One relayed `varagity.backup` log line. */
export interface BackupLogLine {
  level: string;
  message: string;
}

export interface BackupView {
  /** The run snapshot (`null` until the first status frame / when idle). */
  run: BackupRun | null;
  /**
   * The stage the run last reported — `snapshot` | `documents` | `chunks` |
   * `finalize` for an export, `validate` | `load` | `index` | `bm25` for a
   * restore. `null` before the first frame and once terminal.
   */
  stage: string | null;
  /** Rows written so far in a counted stage (`null` otherwise). */
  current: number | null;
  /** The counted stage's row total (`null` otherwise). */
  total: number | null;
  /** Relayed log tail, oldest first (capped). */
  logs: BackupLogLine[];
}

export const initialBackupView: BackupView = {
  run: null,
  stage: null,
  current: null,
  total: null,
  logs: [],
};

/** A backup run logs a handful of lines; the cap only guards a runaway. */
export const MAX_BACKUP_LOG_LINES = 200;

/** Apply one SSE event to the view (pure — returns a new object). */
export function reduceBackupEvent(view: BackupView, event: BackupEvent): BackupView {
  switch (event.type) {
    case "status": {
      const run = event.data.run ?? null;
      // A fresh "running" snapshot begins a new run: drop the previous
      // run's residue (the feed replays each run from scratch).
      const isNewRun =
        run !== null && run.state === "running" && run.run_id !== view.run?.run_id;
      const base = isNewRun ? initialBackupView : view;
      const terminal = run !== null && run.state !== "running";
      return {
        ...base,
        run,
        stage: terminal ? null : base.stage,
        current: terminal ? null : base.current,
        total: terminal ? null : base.total,
      };
    }
    case "progress": {
      const data = event.data;
      // Forward-compatible: a stage this client doesn't know (a later
      // operation's) still moves the activity line, labelled by its name.
      return {
        ...view,
        stage: data.stage,
        current: data.current ?? null,
        total: data.total ?? null,
      };
    }
    case "log": {
      const logs = [...view.logs, { level: event.data.level, message: event.data.message }];
      return { ...view, logs: logs.slice(-MAX_BACKUP_LOG_LINES) };
    }
    default:
      return view;
  }
}

/** Human label for the stage a backup run is in (`null` when idle). */
export function backupStageLabel(view: BackupView): string | null {
  switch (view.stage) {
    case null:
      return null;
    case "snapshot":
      return "taking a consistent snapshot";
    case "documents":
      return "writing documents";
    case "chunks":
      return "writing chunks and vectors";
    case "finalize":
      return "sealing the bundle";
    case "validate":
      return "checking the bundle";
    case "load":
      return "loading rows";
    case "index":
      return "building the vector index";
    case "bm25":
      return "rebuilding the BM25 index";
    default:
      return view.stage;
  }
}

/** The unit a counted stage's `current / total` is measured in. */
export function backupStageUnit(stage: string | null): string {
  switch (stage) {
    case "documents":
      return "documents";
    case "chunks":
      return "chunks";
    case "load":
      return "chunks loaded";
    case "bm25":
      return "chunks indexed";
    default:
      return "rows";
  }
}

/** A restore's four stages, in order, as the panel's step strip names them. */
export const RESTORE_STEPS = [
  { stage: "validate", label: "validate" },
  { stage: "load", label: "load rows" },
  { stage: "index", label: "vector index" },
  { stage: "bm25", label: "BM25 index" },
] as const;

export type RestoreStepState = "done" | "now" | "pending";

/**
 * Where a restore is, step by step: the steps before the reported stage are
 * done, the reported one is in progress, and a completed run has done them
 * all. (A failed run marks nothing done — its error says where it stopped.)
 */
export function restoreSteps(
  view: BackupView,
): { stage: string; label: string; state: RestoreStepState }[] {
  const completed = view.run?.state === "completed";
  const at = RESTORE_STEPS.findIndex((step) => step.stage === view.stage);
  return RESTORE_STEPS.map((step, index) => ({
    stage: step.stage,
    label: step.label,
    state:
      completed || (at !== -1 && index < at) ? "done" : index === at ? "now" : "pending",
  }));
}

/** `58.3 → "58 s"`, `125 → "2 min 5 s"` — a restore's wall clock, readably. */
export function formatDuration(seconds: number): string {
  const whole = Math.round(seconds);
  if (whole < 60) return `${whole} s`;
  const minutes = Math.floor(whole / 60);
  const rest = whole % 60;
  return rest === 0 ? `${minutes} min` : `${minutes} min ${rest} s`;
}

/** The "Restore summary" a completed restore renders, as `[label, value]` rows. */
export function restoreSummaryRows(summary: BackupSummary): [string, string][] {
  const counts = summary.counts;
  const rows: [string, string][] = [];
  if (counts.documents !== undefined) rows.push(["documents", counts.documents.toLocaleString()]);
  if (counts.chunks !== undefined) rows.push(["chunks", counts.chunks.toLocaleString()]);
  if (counts.bm25 !== undefined) rows.push(["BM25 docs", counts.bm25.toLocaleString()]);
  rows.push(["took", formatDuration(summary.wall_clock_s)]);
  return rows;
}

/**
 * The confirm dialog's description: what a restore replaces, and with what.
 *
 * `current` is what the corpus holds now (`null` when unknown — the line
 * then just says "the current corpus").
 */
export function describeReplacement(
  current: { documents: number; chunks: number } | null,
  manifest: BundleManifest | null,
): string {
  const now =
    current === null
      ? "the current corpus"
      : `the ${formatCount(current.documents, "documents")} and ${formatCount(current.chunks, "chunks")}`;
  const incoming =
    manifest === null
      ? "this backup's rows"
      : `this backup's ${Object.entries(manifest.counts)
          .map(([noun, count]) => formatCount(count, noun))
          .join(" and ")}`;
  return `Replaces ${now} in both stores with ${incoming}. Nothing is re-parsed, re-contextualized or re-embedded. Conversations keep their evidence snapshots.`;
}

/**
 * Why restoring a bundle will flag the corpus stale, or `null` when it won't.
 *
 * `drift` is the server's list of reingest-affecting settings the bundle was
 * made under that differ from this stack's; the bundle's own values come
 * from its manifest.
 */
export function describeStaleRestore(bundle: BundleOut): string | null {
  const manifest = bundle.manifest ?? null;
  const parts: string[] = [];
  if (bundle.drift.length > 0) {
    const recorded = manifest?.ingest_settings ?? {};
    const values = bundle.drift.map((name) =>
      recorded[name] === undefined ? `${name} unrecorded` : `${name}=${recorded[name]}`,
    );
    parts.push(
      `It was made under ${values.join(", ")}, which differ from this stack's settings`,
    );
  }
  if (manifest?.stale) parts.push("it was already flagged stale when it was exported");
  if (parts.length === 0) return null;
  return `${parts.join("; ")} — the corpus will be flagged stale until a re-ingest.`;
}

/** Singular nouns for the `counts` keys a manifest carries. */
const SINGULAR: Record<string, string> = {
  documents: "document",
  chunks: "chunk",
};

/** `2 → "2 documents"`, `1 → "1 document"` — locale-grouped digits. */
export function formatCount(count: number, noun: string): string {
  const unit = count === 1 ? (SINGULAR[noun] ?? noun) : noun;
  return `${count.toLocaleString()} ${unit}`;
}

/** `12345678 → "11.8 MB"` — bundles run from kilobytes to gigabytes. */
export function formatBytes(bytes: number): string {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${unit === 0 ? value : value.toFixed(1)} ${units[unit]}`;
}

/** What one bundle holds, and how it was made — the list row's second line. */
export interface BundleDescription {
  /** `"142 documents · 10,233 chunks"`. */
  contents: string;
  /** `"recursive_character 400/50 · contextualized"`, or `null` when unknown. */
  settings: string | null;
}

/** Summarize a bundle's manifest for its row in the backups list. */
export function describeBundle(manifest: BundleManifest): BundleDescription {
  const contents = Object.entries(manifest.counts)
    .map(([noun, count]) => formatCount(count, noun))
    .join(" · ");
  const ingest = manifest.ingest_settings ?? null;
  if (ingest === null) return { contents, settings: null };
  const parts: string[] = [];
  const strategy = ingest.CHUNKING_STRATEGY;
  if (strategy) {
    const size = ingest.CHUNK_SIZE;
    const overlap = ingest.CHUNK_OVERLAP;
    parts.push(size && overlap ? `${strategy} ${size}/${overlap}` : strategy);
  }
  if (ingest.CONTEXTUALIZE === "true") parts.push("contextualized");
  else if (ingest.CONTEXTUALIZE === "false") parts.push("not contextualized");
  return { contents, settings: parts.length > 0 ? parts.join(" · ") : null };
}
