import { describe, expect, it } from "vitest";

import type {
  BackupEvent,
  BackupProgressEvent,
  BackupRun,
  BundleManifest,
} from "@/lib/api";
import {
  backupStageLabel,
  backupStageUnit,
  describeBundle,
  formatBytes,
  formatCount,
  initialBackupView,
  MAX_BACKUP_LOG_LINES,
  reduceBackupEvent,
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
      progress({ stage: "load" as BackupProgressEvent["stage"], current: 3, total: 9 }),
    );
    expect(view.stage).toBe("load");
    expect(backupStageLabel(view)).toBe("load");
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
    expect(formatCount(1, "thread-days")).toBe("1 thread-days"); // unknown nouns pass through
  });

  it("formats byte sizes from bytes to gigabytes", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(1536)).toBe("1.5 KB");
    expect(formatBytes(52_428_800)).toBe("50.0 MB");
    expect(formatBytes(3 * 1024 ** 3)).toBe("3.0 GB");
  });
});
