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
  GraphStatus,
  TargetCorpus,
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
   * `finalize` for a RAG export, `snapshot` | `pack` | `finalize` for a
   * graph export, `validate` | `load` | `index` | `bm25` for a RAG restore,
   * `validate` | `extract` | `swap` | `open` for a graph restore. `null`
   * before the first frame and once terminal.
   */
  stage: string | null;
  /** Rows (`pack`, `extract`: bytes) done so far in a counted stage (`null` otherwise). */
  current: number | null;
  /** The counted stage's row (`pack`, `extract`: byte) total (`null` otherwise). */
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
    case "pack":
      return "packing the graph's files";
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
    case "extract":
      return "extracting the graph's files";
    case "swap":
      return "swapping the restored graph in";
    case "open":
      return "opening the restored graph";
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
    case "pack":
      return "packed";
    case "extract":
      return "extracted";
    default:
      return "rows";
  }
}

/** Stages that count bytes rather than rows — a graph export or restore copies files. */
const BYTE_STAGES: ReadonlySet<string> = new Set(["pack", "extract"]);

/** A counted stage's progress, ready for the panel's bar and its caption. */
export interface BackupProgress {
  /** Done so far, worded — `"6,200"`, or `"182.0 MB"` for a byte-counted stage. */
  done: string;
  /** The stage's total, worded the same way. */
  total: string;
  /** What is being counted (`"chunks loaded"`, `"packed"`). */
  unit: string;
  /** The raw counts, for the progressbar's aria values. */
  current: number;
  max: number;
  /** `current / max`, clamped to 1 (a file that grew mid-copy never overflows). */
  fraction: number;
}

/**
 * Where a counted stage is — rows for every RAG stage, **bytes** for a graph
 * export's `pack` and a graph restore's `extract`, so `182.0 MB / 256.0 MB
 * extracted` rather than a raw byte count. `null` while the stage counts
 * nothing.
 */
export function backupProgress(view: BackupView): BackupProgress | null {
  if (view.total === null || view.total <= 0) return null;
  const current = view.current ?? 0;
  const word = BYTE_STAGES.has(view.stage ?? "")
    ? formatBytes
    : (value: number) => value.toLocaleString();
  return {
    done: word(current),
    total: word(view.total),
    unit: backupStageUnit(view.stage),
    current,
    max: view.total,
    fraction: Math.min(1, current / view.total),
  };
}

/** A document-corpus restore's four stages, in order, as the panel's step strip names them. */
export const RESTORE_STEPS = [
  { stage: "validate", label: "validate" },
  { stage: "load", label: "load rows" },
  { stage: "index", label: "vector index" },
  { stage: "bm25", label: "BM25 index" },
] as const;

/** A graph restore's four stages: extract the files, swap them in, open them. */
export const GRAPH_RESTORE_STEPS = [
  { stage: "validate", label: "validate" },
  { stage: "extract", label: "extract" },
  { stage: "swap", label: "swap" },
  { stage: "open", label: "open" },
] as const;

export type RestoreStepState = "done" | "now" | "pending";

/** The step strip for one corpus's restore. */
export function restoreStepsFor(
  corpus: TargetCorpus,
): readonly { stage: string; label: string }[] {
  return corpus === "graph" ? GRAPH_RESTORE_STEPS : RESTORE_STEPS;
}

/**
 * Where a restore is, step by step — the run's own corpus picks the strip.
 * The steps before the reported stage are done, the reported one is in
 * progress, and a completed run has done them all. (A failed run marks
 * nothing done — its error says where it stopped.)
 */
export function restoreSteps(
  view: BackupView,
): { stage: string; label: string; state: RestoreStepState }[] {
  const steps = restoreStepsFor(view.run?.corpus ?? "rag");
  const completed = view.run?.state === "completed";
  const at = steps.findIndex((step) => step.stage === view.stage);
  return steps.map((step, index) => ({
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

/**
 * The counts a restore summary shows, in order, with their labels — a
 * document restore's rows and BM25 docs, a graph restore's thread-days and
 * sizes. Anything else a summary carries is left out.
 */
const SUMMARY_COUNTS: [key: string, label: string][] = [
  ["documents", "documents"],
  ["chunks", "chunks"],
  ["bm25", "BM25 docs"],
  ["thread_days", "thread-days"],
  ["messages", "messages"],
  ["entities", "entities"],
  ["relations", "relations"],
];

/** The "Restore summary" a completed restore renders, as `[label, value]` rows. */
export function restoreSummaryRows(summary: BackupSummary): [string, string][] {
  const counts = summary.counts;
  const rows: [string, string][] = [];
  for (const [key, label] of SUMMARY_COUNTS) {
    const count = counts[key];
    if (count !== undefined) rows.push([label, count.toLocaleString()]);
  }
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
          .map(([key, count]) => formatCount(count, countNoun(key)))
          .join(" and ")}`;
  return `Replaces ${now} in both stores with ${incoming}. Nothing is re-parsed, re-contextualized or re-embedded. Conversations keep their evidence snapshots.`;
}

/**
 * The graph confirm dialog's description: what a graph restore replaces,
 * and with what. `holding` is what the graph holds now (the panel's "Now:"
 * line; `null` when unknown).
 */
export function describeGraphReplacement(
  holding: string | null,
  manifest: BundleManifest | null,
): string {
  const now = holding === null ? "the current graph" : `the current graph — ${holding} —`;
  const contents = manifest === null ? "" : describeBundle(manifest).contents;
  const incoming = contents === "" ? "this backup's graph" : `this backup's ${contents}`;
  return `Replaces ${now} with ${incoming}. Nothing is re-extracted; graph chat pauses for the few seconds the swap takes.`;
}

/**
 * Why restoring a bundle will flag its corpus stale, or `null` when it won't.
 *
 * `drift` is the server's list of reingest-affecting settings the bundle was
 * made under that differ from this stack's; the bundle's own values come
 * from its manifest. A graph bundle has no drift — it comes back exactly as
 * stale as it was exported.
 */
export function describeStaleRestore(bundle: BundleOut): string | null {
  const manifest = bundle.manifest ?? null;
  if (manifest?.corpus === "graph") {
    return manifest.stale
      ? "It was flagged stale when it was exported (a source had been removed) — the graph will be flagged stale until a re-build."
      : null;
  }
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

/** The line under a finished restore that left its corpus flagged stale. */
export function restoredStaleNote(corpus: TargetCorpus): string {
  return corpus === "graph"
    ? "The backup was already flagged stale when it was exported, so the graph is flagged stale — re-build to apply."
    : "The restored rows don't match this stack's ingest settings (or were already stale when exported), so the corpus is flagged stale — re-ingest to apply them.";
}

/** Singular forms of the plural nouns a count is worded with. */
const SINGULAR: Record<string, string> = {
  documents: "document",
  chunks: "chunk",
  "thread-days": "thread-day",
  messages: "message",
  entities: "entity",
  relations: "relation",
};

/** `counts` keys whose display noun isn't the key itself (the rest read as-is). */
const COUNT_NOUNS: Record<string, string> = {
  thread_days: "thread-days",
};

/** The plural noun a manifest's `counts` key is worded with (`thread_days` → `thread-days`). */
export function countNoun(key: string): string {
  return COUNT_NOUNS[key] ?? key;
}

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
  /** `"142 documents · 10,233 chunks"`, or `"799 thread-days · 64,592 messages"`. */
  contents: string;
  /**
   * `"recursive_character 400/50 · contextualized"` for a RAG bundle, the
   * engine that wrote a graph bundle's files (`"lightrag-hku 1.5.4"`), or
   * `null` when unknown.
   */
  settings: string | null;
}

/** Summarize a bundle's manifest for its row in the backups list. */
export function describeBundle(manifest: BundleManifest): BundleDescription {
  const contents = Object.entries(manifest.counts)
    .map(([key, count]) => formatCount(count, countNoun(key)))
    .join(" · ");
  // A graph bundle is its engine's files verbatim, so what it was "made
  // under" is that engine's version — the thing a restore holds it to.
  const engine = manifest.engine ?? null;
  if (engine !== null) return { contents, settings: `${engine.package} ${engine.version}` };
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

/**
 * What the graph holds right now, for its panel's "Now:" line —
 * `"799 thread-days · 64,592 messages · 5,000 entities · 6,680 relations"`,
 * straight off the status poll (the engine's document counts, all statuses,
 * and the workdir's size summary). Figures the graph would not report are
 * left out; `null` when it reports nothing at all.
 */
export function describeGraphHolding(status: GraphStatus): string | null {
  const threadDays = Object.values(status.documents).reduce((sum, count) => sum + count, 0);
  const parts: string[] = [];
  if (threadDays > 0) parts.push(formatCount(threadDays, "thread-days"));
  if (status.message_guids != null) parts.push(formatCount(status.message_guids, "messages"));
  if (status.entities != null) parts.push(formatCount(status.entities, "entities"));
  if (status.relations != null) parts.push(formatCount(status.relations, "relations"));
  return parts.length > 0 ? parts.join(" · ") : null;
}

/**
 * Whether the graph is confirmed unbuilt — no file in its working directory,
 * so nothing to export and nothing a restore would replace. The server's
 * own `nothing_to_export` rule (`workdir_files`), not the size figures: a
 * graph that won't open, or a first build killed before it wrote its
 * summary, reports no sizes but still has files to back up — and to replace.
 */
export function graphIsEmpty(status: GraphStatus): boolean {
  return status.workdir_files === 0;
}
