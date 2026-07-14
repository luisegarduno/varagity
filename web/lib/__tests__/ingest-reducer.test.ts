import { describe, expect, it } from "vitest";

import type { IngestEvent, IngestRun } from "@/lib/api";
import {
  initialIngestView,
  MAX_LOG_LINES,
  reduceIngestEvent,
  type IngestView,
} from "@/lib/ingest-reducer";

function run(state: string, overrides?: Partial<IngestRun>): IngestRun {
  return {
    run_id: "r1",
    state,
    reingest: false,
    started_at: "2026-07-13T10:00:00Z",
    finished_at: null,
    summary: null,
    error: null,
    ...overrides,
  };
}

function reduceAll(events: IngestEvent[], from: IngestView = initialIngestView): IngestView {
  return events.reduce(reduceIngestEvent, from);
}

/** The feed a 2-file run emits, as the runner orders it. */
function fullRunEvents(): IngestEvent[] {
  return [
    { type: "status", data: { run: run("running") } },
    { type: "progress", data: { stage: "discover", total: 2 } },
    { type: "progress", data: { stage: "parse", file: "a.txt" } },
    { type: "progress", data: { stage: "chunk", file: "a.txt", total: 2 } },
    { type: "progress", data: { stage: "contextualize", file: "a.txt", current: 0, total: 2 } },
    { type: "progress", data: { stage: "contextualize", file: "a.txt", current: 1, total: 2 } },
    { type: "progress", data: { stage: "contextualize", file: "a.txt", current: 2, total: 2 } },
    { type: "progress", data: { stage: "embed", file: "a.txt", total: 2 } },
    { type: "progress", data: { stage: "store", file: "a.txt", total: 2 } },
    {
      type: "progress",
      data: {
        stage: "file_done",
        file: "a.txt",
        outcome: "ingested",
        total: 2,
        files_done: 1,
        files_total: 2,
      },
    },
    { type: "log", data: { level: "INFO", message: "b.txt: unchanged — skipping" } },
    {
      type: "progress",
      data: {
        stage: "file_done",
        file: "b.txt",
        outcome: "skipped",
        files_done: 2,
        files_total: 2,
      },
    },
    {
      type: "status",
      data: {
        run: run("completed", {
          finished_at: "2026-07-13T10:05:00Z",
          summary: {
            discovered: 2,
            ingested: 1,
            skipped: 1,
            no_text: 0,
            unsupported: 0,
            failed: 0,
            chunks: 2,
          },
        }),
      },
    },
  ];
}

describe("reduceIngestEvent", () => {
  it("an idle status frame leaves the view idle", () => {
    const view = reduceIngestEvent(initialIngestView, {
      type: "status",
      data: { run: null },
    });
    expect(view.run).toBeNull();
    expect(view.filesTotal).toBeNull();
  });

  it("replay of a full run rebuilds the terminal picture", () => {
    const view = reduceAll(fullRunEvents());
    expect(view.run?.state).toBe("completed");
    expect(view.run?.summary?.ingested).toBe(1);
    expect(view.filesTotal).toBe(2);
    expect(view.filesDone).toBe(2);
    expect(view.currentFile).toBeNull(); // terminal clears the activity line
    expect(view.currentStage).toBeNull();
    expect(view.contextualize).toBeNull();
    expect(view.outcomes).toEqual([
      { file: "a.txt", outcome: "ingested", chunks: 2 },
      { file: "b.txt", outcome: "skipped", chunks: null },
    ]);
    expect(view.logs).toEqual([{ level: "INFO", message: "b.txt: unchanged — skipping" }]);
  });

  it("mid-run state tracks the current file, stage, and contextualize ticks", () => {
    const events = fullRunEvents().slice(0, 6); // through the first tick
    const view = reduceAll(events);
    expect(view.run?.state).toBe("running");
    expect(view.currentFile).toBe("a.txt");
    expect(view.currentStage).toBe("contextualize");
    expect(view.contextualize).toEqual({ done: 1, total: 2 });
    expect(view.filesDone).toBe(0);
  });

  it("file_done advances the counters and clears the activity line", () => {
    const events = fullRunEvents().slice(0, 10); // through a.txt's file_done
    const view = reduceAll(events);
    expect(view.filesDone).toBe(1);
    expect(view.filesTotal).toBe(2);
    expect(view.currentFile).toBeNull();
    expect(view.contextualize).toBeNull();
  });

  it("a new running run resets a previous run's residue", () => {
    const finished = reduceAll(fullRunEvents());
    const fresh = reduceIngestEvent(finished, {
      type: "status",
      data: { run: run("running", { run_id: "r2" }) },
    });
    expect(fresh.run?.run_id).toBe("r2");
    expect(fresh.filesDone).toBe(0);
    expect(fresh.logs).toEqual([]);
    expect(fresh.outcomes).toEqual([]);
  });

  it("a failed run carries its error and clears the activity line", () => {
    const view = reduceAll([
      { type: "status", data: { run: run("running") } },
      { type: "progress", data: { stage: "parse", file: "a.txt" } },
      { type: "status", data: { run: run("failed", { error: "RuntimeError: es fell over" }) } },
    ]);
    expect(view.run?.state).toBe("failed");
    expect(view.run?.error).toContain("es fell over");
    expect(view.currentFile).toBeNull();
  });

  it("caps the log tail", () => {
    const lines: IngestEvent[] = Array.from({ length: MAX_LOG_LINES + 25 }, (_, i) => ({
      type: "log",
      data: { level: "INFO", message: `line ${i}` },
    }));
    const view = reduceAll(lines);
    expect(view.logs).toHaveLength(MAX_LOG_LINES);
    expect(view.logs[0].message).toBe("line 25"); // oldest dropped
  });

  it("ignores unknown progress stages (forward compatibility)", () => {
    const view = reduceIngestEvent(initialIngestView, {
      type: "progress",
      data: { stage: "quantum_dedupe" },
    });
    expect(view).toEqual(initialIngestView);
  });

  it("is pure — inputs are never mutated", () => {
    const before = reduceAll(fullRunEvents().slice(0, 3));
    const frozen = JSON.parse(JSON.stringify(before)) as IngestView;
    reduceIngestEvent(before, {
      type: "log",
      data: { level: "INFO", message: "x" },
    });
    expect(before).toEqual(frozen);
  });
});
