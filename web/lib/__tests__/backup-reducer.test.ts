import { describe, expect, it } from "vitest";

import type {
  BackupEvent,
  BackupProgressEvent,
  BackupRun,
  BundleManifest,
  BundleOut,
  GraphStatus,
} from "@/lib/api";
import {
  backupProgress,
  backupStageLabel,
  backupStageUnit,
  countNoun,
  describeGraphHolding,
  describeBundle,
  describeReplacement,
  describeStaleRestore,
  formatBytes,
  formatCount,
  formatDuration,
  graphIsEmpty,
  initialBackupView,
  MAX_BACKUP_LOG_LINES,
  reduceBackupEvent,
  RESTORE_STEPS,
  restoreSteps,
  restoreSummaryRows,
  type BackupView,
} from "@/lib/backup-reducer";

function run(state: BackupRun["state"], overrides?: Partial<BackupRun>): BackupRun {
  return {
    run_id: "b1",
    corpus: "rag",
    operation: "export",
    state,
    bundle: null,
    saved_to: null,
    started_at: "2026-11-08T13:12:00Z",
    finished_at: null,
    error: null,
    summary: null,
    ...overrides,
  };
}

/** A full progress frame: the wire shape carries every key, nulls included. */
function progress(
  data: Partial<BackupProgressEvent> & Pick<BackupProgressEvent, "stage">,
): BackupEvent {
  return { type: "progress", data: { current: null, total: null, ...data } };
}

function reduceAll(events: BackupEvent[], from: BackupView = initialBackupView): BackupView {
  return events.reduce(reduceBackupEvent, from);
}

const NAME = "varagity-rag-20261108-1312.zip";

/** The feed one export emits, as the runner orders it. */
function fullRunEvents(): BackupEvent[] {
  return [
    { type: "status", data: { run: run("running") } },
    progress({ stage: "snapshot" }),
    { type: "log", data: { level: "INFO", message: "exporting 2 document(s) and 5 chunk(s)" } },
    progress({ stage: "documents", current: 0, total: 2 }),
    progress({ stage: "documents", current: 2, total: 2 }),
    progress({ stage: "chunks", current: 0, total: 5 }),
    progress({ stage: "chunks", current: 5, total: 5 }),
    progress({ stage: "finalize" }),
    {
      type: "status",
      data: {
        run: run("completed", {
          bundle: NAME,
          saved_to: `docs/varagity-exports/${NAME}`,
          finished_at: "2026-11-08T13:12:04Z",
          summary: { counts: { documents: 2, chunks: 5 }, size_bytes: 52_428_800, wall_clock_s: 3.2 },
        }),
      },
    },
  ];
}

describe("reduceBackupEvent", () => {
  it("an idle status frame leaves the view idle", () => {
    const view = reduceBackupEvent(initialBackupView, { type: "status", data: { run: null } });
    expect(view.run).toBeNull();
    expect(view.stage).toBeNull();
  });

  it("replay of a full export rebuilds the terminal picture", () => {
    const view = reduceAll(fullRunEvents());
    expect(view.run?.state).toBe("completed");
    expect(view.run?.saved_to).toBe(`docs/varagity-exports/${NAME}`);
    expect(view.run?.summary?.counts).toEqual({ documents: 2, chunks: 5 });
    // A terminal frame clears the activity line.
    expect(view.stage).toBeNull();
    expect(view.current).toBeNull();
    expect(view.total).toBeNull();
    expect(view.logs).toEqual([
      { level: "INFO", message: "exporting 2 document(s) and 5 chunk(s)" },
    ]);
  });

  it("mid-run state tracks the stage and its row counts", () => {
    const view = reduceAll(fullRunEvents().slice(0, 6));
    expect(view.run?.state).toBe("running");
    expect(view.stage).toBe("chunks");
    expect(view.current).toBe(0);
    expect(view.total).toBe(5);
  });

  it("an uncounted stage clears the previous stage's counts", () => {
    const view = reduceAll(fullRunEvents().slice(0, 8));
    expect(view.stage).toBe("finalize");
    expect(view.current).toBeNull();
    expect(view.total).toBeNull();
  });

  it("a failed run keeps its error and drops the activity line", () => {
    const view = reduceAll([
      ...fullRunEvents().slice(0, 4),
      { type: "status", data: { run: run("failed", { error: "OSError: disk full" }) } },
    ]);
    expect(view.run?.state).toBe("failed");
    expect(view.run?.error).toBe("OSError: disk full");
    expect(view.run?.bundle).toBeNull();
    expect(view.stage).toBeNull();
    expect(view.logs).toHaveLength(1); // the log tail survives the failure
  });

  it("a new run's first frame drops the previous run's residue", () => {
    const previous = reduceAll(fullRunEvents());
    const next = reduceBackupEvent(previous, {
      type: "status",
      data: { run: run("running", { run_id: "b2" }) },
    });
    expect(next.run?.run_id).toBe("b2");
    expect(next.logs).toEqual([]);
    expect(next.stage).toBeNull();
  });

  it("re-replaying the same run's first frame keeps its state", () => {
    const mid = reduceAll(fullRunEvents().slice(0, 4));
    const again = reduceBackupEvent(mid, { type: "status", data: { run: run("running") } });
    expect(again.logs).toEqual(mid.logs);
    expect(again.stage).toBe(mid.stage);
  });

  it("replays to the same view however the feed is split", () => {
    const events = fullRunEvents();
    const whole = reduceAll(events);
    const resumed = reduceAll(events.slice(4), reduceAll(events.slice(0, 4)));
    expect(resumed).toEqual(whole);
  });

  it("an unknown stage still moves the activity line (forward-compatible)", () => {
    const view = reduceBackupEvent(
      reduceAll(fullRunEvents().slice(0, 1)),
      progress({ stage: "defragment" as BackupProgressEvent["stage"], current: 3, total: 9 }),
    );
    expect(view.stage).toBe("defragment");
    expect(backupStageLabel(view)).toBe("defragment");
  });

  it("caps the log tail", () => {
    const logs: BackupEvent[] = Array.from({ length: MAX_BACKUP_LOG_LINES + 5 }, (_, i) => ({
      type: "log",
      data: { level: "INFO", message: `line ${i}` },
    }));
    const view = reduceAll(logs);
    expect(view.logs).toHaveLength(MAX_BACKUP_LOG_LINES);
    expect(view.logs[0].message).toBe("line 5");
  });

  it("is pure: the input view is never mutated", () => {
    const before = reduceAll(fullRunEvents().slice(0, 3));
    const snapshot = structuredClone(before);
    for (const event of fullRunEvents()) reduceBackupEvent(before, event);
    expect(before).toEqual(snapshot);
  });

  it("ignores an event type it does not know", () => {
    const view = reduceAll(fullRunEvents().slice(0, 2));
    const unknown = { type: "mystery", data: {} } as unknown as BackupEvent;
    expect(reduceBackupEvent(view, unknown)).toBe(view);
  });
});

describe("backup stage presentation", () => {
  it("labels every export stage, and nothing when idle", () => {
    const at = (stage: string | null) => backupStageLabel({ ...initialBackupView, stage });
    expect(at(null)).toBeNull();
    expect(at("snapshot")).toBe("taking a consistent snapshot");
    expect(at("documents")).toBe("writing documents");
    expect(at("chunks")).toBe("writing chunks and vectors");
    expect(at("finalize")).toBe("sealing the bundle");
  });

  it("names the unit a counted stage is measured in", () => {
    expect(backupStageUnit("documents")).toBe("documents");
    expect(backupStageUnit("chunks")).toBe("chunks");
    expect(backupStageUnit(null)).toBe("rows");
  });
});

function manifest(overrides?: Partial<BundleManifest>): BundleManifest {
  return {
    format: "varagity-backup",
    version: 1,
    corpus: "rag",
    created_at: "2026-11-08T13:12:00Z",
    embedding_model: "infloat/multilingual-e5-large-instruct",
    embedding_dim: 1024,
    stale: false,
    counts: { documents: 142, chunks: 10_233 },
    ingest_settings: {
      CHUNKING_STRATEGY: "markdown_aware",
      CHUNK_OVERLAP: "100",
      CHUNK_SIZE: "800",
      CONTEXTUALIZE: "true",
      OCR_ENGINE: "easyocr",
    },
    members: [],
    ...overrides,
  };
}

describe("describeBundle", () => {
  it("summarizes the contents and the settings the rows were made under", () => {
    expect(describeBundle(manifest())).toEqual({
      contents: `142 documents · ${(10_233).toLocaleString()} chunks`,
      settings: "markdown_aware 800/100 · contextualized",
    });
  });

  it("says when the corpus was not contextualized", () => {
    const settings = { CHUNKING_STRATEGY: "semantic", CONTEXTUALIZE: "false" };
    expect(describeBundle(manifest({ ingest_settings: settings })).settings).toBe(
      "semantic · not contextualized",
    );
  });

  it("has no settings line without recorded settings", () => {
    expect(describeBundle(manifest({ ingest_settings: null })).settings).toBeNull();
    expect(describeBundle(manifest({ ingest_settings: {} })).settings).toBeNull();
  });

  it("uses the singular for one", () => {
    expect(describeBundle(manifest({ counts: { documents: 1, chunks: 1 } })).contents).toBe(
      "1 document · 1 chunk",
    );
  });
});

describe("formatting", () => {
  it("formats counts with a noun", () => {
    expect(formatCount(0, "documents")).toBe("0 documents");
    expect(formatCount(1, "chunks")).toBe("1 chunk");
    expect(formatCount(1, "widgets")).toBe("1 widgets"); // unknown nouns pass through
  });

  it("formats byte sizes from bytes to gigabytes", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(1536)).toBe("1.5 KB");
    expect(formatBytes(52_428_800)).toBe("50.0 MB");
    expect(formatBytes(3 * 1024 ** 3)).toBe("3.0 GB");
  });
});

// ── restore (phase 2) ──────────────────────────────────────────────────

const RESTORING = "varagity-rag-20261019-0904.zip";

function restoreRun(state: BackupRun["state"], overrides?: Partial<BackupRun>): BackupRun {
  return run(state, { run_id: "r1", operation: "restore", bundle: RESTORING, ...overrides });
}

/** The feed one restore emits, as the runner orders it. */
function fullRestoreEvents(): BackupEvent[] {
  return [
    { type: "status", data: { run: restoreRun("running") } },
    progress({ stage: "validate" }),
    { type: "log", data: { level: "INFO", message: `restoring 128 document(s) and 8977 chunk(s) from ${RESTORING}` } },
    progress({ stage: "load", current: 0, total: 8977 }),
    progress({ stage: "load", current: 6200, total: 8977 }),
    progress({ stage: "load", current: 8977, total: 8977 }),
    progress({ stage: "index" }),
    progress({ stage: "bm25", current: 0, total: 8977 }),
    progress({ stage: "bm25", current: 8977, total: 8977 }),
    {
      type: "status",
      data: {
        run: restoreRun("completed", {
          finished_at: "2026-11-08T13:13:00Z",
          summary: {
            counts: { documents: 128, chunks: 8977, bm25: 8977 },
            size_bytes: 45_822_976,
            wall_clock_s: 52.4,
            stale: true,
          },
        }),
      },
    },
  ];
}

describe("restore replays", () => {
  it("a full restore rebuilds the terminal picture, every step done", () => {
    const view = reduceAll(fullRestoreEvents());
    expect(view.run?.operation).toBe("restore");
    expect(view.run?.bundle).toBe(RESTORING);
    expect(view.run?.summary?.stale).toBe(true);
    expect(view.stage).toBeNull();
    expect(restoreSteps(view).map((step) => step.state)).toEqual([
      "done",
      "done",
      "done",
      "done",
    ]);
  });

  it("mid-load: validate done, load in progress with its chunk counts", () => {
    const view = reduceAll(fullRestoreEvents().slice(0, 5));
    expect(view.stage).toBe("load");
    expect([view.current, view.total]).toEqual([6200, 8977]);
    expect(backupStageLabel(view)).toBe("loading rows");
    expect(backupStageUnit(view.stage)).toBe("chunks loaded");
    expect(restoreSteps(view).map((step) => step.state)).toEqual([
      "done",
      "now",
      "pending",
      "pending",
    ]);
  });

  it("the index build is uncounted; the BM25 rebuild counts chunks", () => {
    const indexing = reduceAll(fullRestoreEvents().slice(0, 7));
    expect(indexing.stage).toBe("index");
    expect(indexing.total).toBeNull();
    expect(backupStageLabel(indexing)).toBe("building the vector index");
    const rebuilding = reduceAll(fullRestoreEvents().slice(0, 9));
    expect(backupStageLabel(rebuilding)).toBe("rebuilding the BM25 index");
    expect(backupStageUnit(rebuilding.stage)).toBe("chunks indexed");
    expect(restoreSteps(rebuilding)[3].state).toBe("now");
  });

  it("a failed restore marks nothing done and keeps its error", () => {
    const view = reduceAll([
      ...fullRestoreEvents().slice(0, 9),
      {
        type: "status",
        data: {
          run: restoreRun("failed", {
            error: "BM25RebuildFailed: the rows are restored, but the BM25 index could not be rebuilt",
          }),
        },
      },
    ]);
    expect(view.run?.state).toBe("failed");
    expect(view.run?.error).toMatch(/^BM25RebuildFailed/);
    expect(restoreSteps(view).every((step) => step.state === "pending")).toBe(true);
  });

  it("replays to the same view however the feed is split", () => {
    const events = fullRestoreEvents();
    expect(reduceAll(events.slice(5), reduceAll(events.slice(0, 5)))).toEqual(reduceAll(events));
  });

  it("an export after a restore starts from scratch", () => {
    const restored = reduceAll(fullRestoreEvents());
    const next = reduceBackupEvent(restored, {
      type: "status",
      data: { run: run("running", { run_id: "e2" }) },
    });
    expect(next.run?.operation).toBe("export");
    expect(next.logs).toEqual([]);
  });

  it("labels the validate step", () => {
    expect(backupStageLabel({ ...initialBackupView, stage: "validate" })).toBe(
      "checking the bundle",
    );
    expect(RESTORE_STEPS.map((step) => step.label)).toEqual([
      "validate",
      "load rows",
      "vector index",
      "BM25 index",
    ]);
  });
});

describe("restore presentation", () => {
  it("summarizes a completed restore", () => {
    const view = reduceAll(fullRestoreEvents());
    expect(restoreSummaryRows(view.run!.summary!)).toEqual([
      ["documents", "128"],
      ["chunks", (8977).toLocaleString()],
      ["BM25 docs", (8977).toLocaleString()],
      ["took", "52 s"],
    ]);
    expect(
      restoreSummaryRows({ counts: {}, size_bytes: 0, wall_clock_s: 0.4 }),
    ).toEqual([["took", "0 s"]]);
  });

  it("formats durations", () => {
    expect(formatDuration(58.3)).toBe("58 s");
    expect(formatDuration(60)).toBe("1 min");
    expect(formatDuration(125)).toBe("2 min 5 s");
  });

  it("says what a restore replaces, and with what", () => {
    expect(describeReplacement({ documents: 142, chunks: 10_233 }, manifest())).toBe(
      `Replaces the 142 documents and ${(10_233).toLocaleString()} chunks in both stores with this backup's 142 documents and ${(10_233).toLocaleString()} chunks. Nothing is re-parsed, re-contextualized or re-embedded. Conversations keep their evidence snapshots.`,
    );
    expect(describeReplacement(null, null)).toMatch(
      /^Replaces the current corpus in both stores with this backup's rows\./,
    );
    expect(
      describeReplacement({ documents: 1, chunks: 1 }, manifest({ counts: { documents: 1, chunks: 1 } })),
    ).toMatch(/^Replaces the 1 document and 1 chunk in both stores with this backup's 1 document and 1 chunk\./);
  });

  function listed(overrides?: Partial<BundleOut>): BundleOut {
    return {
      name: RESTORING,
      size_bytes: 45_822_976,
      modified_at: "2026-10-19T09:04:00Z",
      manifest: manifest(),
      restorable: true,
      reason: null,
      drift: [],
      ...overrides,
    };
  }

  it("has nothing to warn about for a bundle that matches the stack", () => {
    expect(describeStaleRestore(listed())).toBeNull();
  });

  it("names the drifted settings with the bundle's own values", () => {
    expect(describeStaleRestore(listed({ drift: ["CHUNK_SIZE", "CHUNKING_STRATEGY"] }))).toBe(
      "It was made under CHUNK_SIZE=800, CHUNKING_STRATEGY=markdown_aware, which differ from this stack's settings — the corpus will be flagged stale until a re-ingest.",
    );
    expect(
      describeStaleRestore(listed({ drift: ["NEW_KNOB"], manifest: manifest({ ingest_settings: null }) })),
    ).toBe(
      "It was made under NEW_KNOB unrecorded, which differ from this stack's settings — the corpus will be flagged stale until a re-ingest.",
    );
  });

  it("warns when the bundle was already stale at export", () => {
    expect(describeStaleRestore(listed({ manifest: manifest({ stale: true }) }))).toBe(
      "it was already flagged stale when it was exported — the corpus will be flagged stale until a re-ingest.",
    );
  });
});

// ── the graph export (phase 3) ─────────────────────────────────────────

const GRAPH_NAME = "varagity-graph-20261108-1315.zip";
const MB = 1024 * 1024;

function graphRun(state: BackupRun["state"], overrides?: Partial<BackupRun>): BackupRun {
  return run(state, { run_id: "g1", corpus: "graph", ...overrides });
}

/** The feed one graph export emits: snapshot → pack (bytes) → finalize. */
function fullGraphExportEvents(): BackupEvent[] {
  return [
    { type: "status", data: { run: graphRun("running") } },
    progress({ stage: "snapshot" }),
    {
      type: "log",
      data: { level: "INFO", message: "exporting 14 graph file(s), 268,435,456 bytes, written by lightrag-hku 1.5.4" },
    },
    progress({ stage: "pack", current: 0, total: 256 * MB }),
    progress({ stage: "pack", current: 182 * MB, total: 256 * MB }),
    progress({ stage: "pack", current: 256 * MB, total: 256 * MB }),
    progress({ stage: "finalize" }),
    {
      type: "status",
      data: {
        run: graphRun("completed", {
          bundle: GRAPH_NAME,
          saved_to: `graph-docs/varagity-exports/${GRAPH_NAME}`,
          finished_at: "2026-11-08T13:15:21Z",
          summary: {
            counts: { thread_days: 799, messages: 64_592, entities: 5000, relations: 6680 },
            size_bytes: 152 * MB,
            wall_clock_s: 21.4,
          },
        }),
      },
    },
  ];
}

describe("graph export replays", () => {
  it("a full graph export rebuilds the terminal picture", () => {
    const view = reduceAll(fullGraphExportEvents());
    expect(view.run?.state).toBe("completed");
    expect(view.run?.corpus).toBe("graph");
    expect(view.run?.saved_to).toBe(`graph-docs/varagity-exports/${GRAPH_NAME}`);
    expect(view.stage).toBeNull();
    expect(backupProgress(view)).toBeNull(); // terminal: no bar
    expect(view.logs).toHaveLength(1);
  });

  it("mid-pack: the bar counts bytes, worded in megabytes", () => {
    const view = reduceAll(fullGraphExportEvents().slice(0, 5));
    expect(view.stage).toBe("pack");
    expect(backupStageLabel(view)).toBe("packing the graph's files");
    expect(backupStageUnit(view.stage)).toBe("packed");
    expect(backupProgress(view)).toEqual({
      done: "182.0 MB",
      total: "256.0 MB",
      unit: "packed",
      current: 182 * MB,
      max: 256 * MB,
      fraction: 182 / 256,
    });
  });

  it("replays to the same view however the feed is split", () => {
    const events = fullGraphExportEvents();
    const whole = reduceAll(events);
    const joinedLate = reduceAll(events.slice(4), reduceAll(events.slice(0, 4)));
    expect(joinedLate).toEqual(whole);
  });

  it("a graph export after a RAG run's frames starts from scratch", () => {
    const afterRag = reduceAll(fullGraphExportEvents(), reduceAll(fullRunEvents()));
    expect(afterRag).toEqual(reduceAll(fullGraphExportEvents()));
  });
});

describe("backupProgress", () => {
  it("words row-counted stages as locale-grouped counts", () => {
    const view = { ...initialBackupView, stage: "load", current: 6200, total: 8977 };
    expect(backupProgress(view)).toMatchObject({
      done: (6200).toLocaleString(),
      total: (8977).toLocaleString(),
      unit: "chunks loaded",
    });
  });

  it("is nothing for an uncounted stage", () => {
    expect(backupProgress({ ...initialBackupView, stage: "index" })).toBeNull();
    expect(backupProgress({ ...initialBackupView, stage: "pack", current: 0, total: 0 })).toBeNull();
  });

  it("never overflows when a file grew since it was listed", () => {
    const view = { ...initialBackupView, stage: "pack", current: 300, total: 200 };
    expect(backupProgress(view)?.fraction).toBe(1);
  });
});

function graphManifest(overrides?: Partial<BundleManifest>): BundleManifest {
  return manifest({
    corpus: "graph",
    counts: { thread_days: 799, messages: 64_592, entities: 5000, relations: 6680 },
    ingest_settings: null,
    engine: { name: "lightrag", package: "lightrag-hku", version: "1.5.4" },
    ...overrides,
  });
}

describe("describeBundle for a graph bundle", () => {
  it("words the graph's counts and names the engine that wrote the files", () => {
    expect(describeBundle(graphManifest())).toEqual({
      contents: `799 thread-days · ${(64_592).toLocaleString()} messages · ${(5000).toLocaleString()} entities · ${(6680).toLocaleString()} relations`,
      settings: "lightrag-hku 1.5.4",
    });
  });

  it("leaves out what the graph never reported", () => {
    expect(describeBundle(graphManifest({ counts: { thread_days: 1, messages: 1 } })).contents).toBe(
      "1 thread-day · 1 message",
    );
    expect(describeBundle(graphManifest({ counts: {} })).contents).toBe("");
  });

  it("maps manifest keys to the nouns the UI uses", () => {
    expect(countNoun("thread_days")).toBe("thread-days");
    expect(countNoun("documents")).toBe("documents");
    expect(formatCount(1, "entities")).toBe("1 entity");
    expect(formatCount(1, "relations")).toBe("1 relation");
  });
});

function status(overrides?: Partial<GraphStatus>): GraphStatus {
  return { enabled: true, stale: false, building: false, documents: {}, ...overrides };
}

describe("the graph's holding", () => {
  it("adds every status's thread-days and the summary's sizes", () => {
    const built = status({
      documents: { processed: 797, failed: 2 },
      message_guids: 64_592,
      entities: 5000,
      relations: 6680,
    });
    expect(describeGraphHolding(built)).toBe(
      `799 thread-days · ${(64_592).toLocaleString()} messages · ${(5000).toLocaleString()} entities · ${(6680).toLocaleString()} relations`,
    );
    expect(graphIsEmpty(built)).toBe(false);
  });

  it("skips what the graph would not say", () => {
    // A first build killed partway: statuses on disk, no summary yet.
    expect(describeGraphHolding(status({ documents: { pending: 3 } }))).toBe("3 thread-days");
  });

  it("an unbuilt graph holds nothing — nothing to export", () => {
    expect(describeGraphHolding(status())).toBeNull();
    expect(graphIsEmpty(status())).toBe(true);
    expect(graphIsEmpty(status({ entities: null, relations: null, message_guids: null }))).toBe(true);
  });
});
