import { describe, expect, it } from "vitest";

import type {
  GraphBuildEvent,
  GraphBuildProgressEvent,
  GraphBuildRun,
} from "@/lib/api";
import {
  graphStageLabel,
  initialGraphBuildView,
  MAX_GRAPH_LOG_LINES,
  reduceGraphBuildEvent,
  type GraphBuildView,
} from "@/lib/graph-build-reducer";

function run(state: string, overrides?: Partial<GraphBuildRun>): GraphBuildRun {
  return {
    run_id: "g1",
    state,
    reingest: false,
    bounded: false,
    message_limit: null,
    since: null,
    started_at: "2026-08-07T10:00:00Z",
    finished_at: null,
    summary: null,
    error: null,
    ...overrides,
  };
}

/** A full progress frame: the wire shape carries every key, nulls included. */
function progress(
  data: Partial<GraphBuildProgressEvent> &
    Pick<GraphBuildProgressEvent, "stage">,
): GraphBuildEvent {
  return {
    type: "progress",
    data: {
      file: null,
      current: null,
      total: null,
      docs_done: null,
      docs_total: null,
      ...data,
    },
  };
}

function reduceAll(
  events: GraphBuildEvent[],
  from: GraphBuildView = initialGraphBuildView,
): GraphBuildView {
  return events.reduce(reduceGraphBuildEvent, from);
}

/** The feed a bounded 2-file run emits, as the runner orders it. */
function boundedRunEvents(): GraphBuildEvent[] {
  return [
    { type: "status", data: { run: run("running", { bounded: true, message_limit: 60 }) } },
    progress({ stage: "scan", total: 3 }),
    progress({ stage: "parse", file: "sms.db", current: 1, total: 3 }),
    progress({ stage: "parse", file: "full.db", current: 2, total: 3 }),
    progress({ stage: "bound", current: 60, total: 30_542 }),
    progress({ stage: "index", total: 60 }),
    progress({ stage: "process", docs_done: 0, docs_total: 9 }),
    {
      type: "log",
      data: { level: "INFO", message: "9 new, 0 changed, 0 unchanged" },
    },
    progress({ stage: "process", docs_done: 9, docs_total: 9 }),
    {
      type: "status",
      data: {
        run: run("completed", {
          bounded: true,
          message_limit: 60,
          finished_at: "2026-08-07T10:13:42Z",
          summary: {
            files_scanned: 3,
            files_parsed: 2,
            files_failed: 0,
            messages_parsed: 30_542,
            messages_indexed: 60,
            messages_seen: 60,
            wall_clock_s: 822,
            resumed: 0,
            documents: { processed: 9 },
            failures: [],
          },
        }),
      },
    },
  ];
}

describe("reduceGraphBuildEvent", () => {
  it("an idle status frame leaves the view idle", () => {
    const view = reduceGraphBuildEvent(initialGraphBuildView, {
      type: "status",
      data: { run: null },
    });
    expect(view.run).toBeNull();
    expect(view.docsTotal).toBeNull();
  });

  it("replay of a full run rebuilds the terminal picture", () => {
    const view = reduceAll(boundedRunEvents());
    expect(view.run?.state).toBe("completed");
    expect(view.run?.summary?.messages_indexed).toBe(60);
    expect(view.filesTotal).toBe(3);
    expect(view.filesDone).toBe(2); // the third file is a sidecar — never parsed
    expect(view.messagesParsed).toBe(30_542);
    expect(view.messagesIndexed).toBe(60);
    expect(view.docsDone).toBe(9);
    expect(view.docsTotal).toBe(9);
    expect(view.stage).toBeNull(); // terminal clears the activity line
    expect(view.currentFile).toBeNull();
    expect(view.logs).toEqual([
      { level: "INFO", message: "9 new, 0 changed, 0 unchanged" },
    ]);
  });

  it("a completed summary corrects a progress bar the sampled ticks left short", () => {
    const view = reduceAll([
      { type: "status", data: { run: run("running") } },
      progress({ stage: "process", docs_done: 9, docs_total: 10 }),
      {
        type: "status",
        data: {
          run: run("completed", {
            finished_at: "2026-08-07T10:13:42Z",
            summary: {
              files_scanned: 1,
              files_parsed: 1,
              files_failed: 0,
              messages_parsed: 226,
              messages_indexed: 226,
              messages_seen: 226,
              wall_clock_s: 1724,
              resumed: 0,
              documents: { processed: 10, failed: 0 },
              failures: [],
            },
          }),
        },
      },
    ]);
    expect(view.docsDone).toBe(10);
    expect(view.docsTotal).toBe(10);
  });

  it("a failed run keeps its honestly-partial document counts", () => {
    const view = reduceAll([
      { type: "status", data: { run: run("running") } },
      progress({ stage: "process", docs_done: 4, docs_total: 10 }),
      { type: "status", data: { run: run("failed", { error: "boom" }) } },
    ]);
    expect(view.docsDone).toBe(4);
    expect(view.docsTotal).toBe(10);
  });

  it("tracks the corpus walk while it runs", () => {
    const view = reduceAll(boundedRunEvents().slice(0, 3));
    expect(view.stage).toBe("parse");
    expect(view.currentFile).toBe("sms.db");
    expect(view.filesDone).toBe(1);
    expect(view.docsTotal).toBeNull(); // the engine clock hasn't started
  });

  it("hands off to the engine clock at index, clearing the file line", () => {
    const view = reduceAll(boundedRunEvents().slice(0, 6));
    expect(view.stage).toBe("index");
    expect(view.currentFile).toBeNull();
    expect(view.messagesIndexed).toBe(60);
  });

  it("an unbounded run learns its message count from index alone", () => {
    // No `bound` frame is emitted for an unbounded run — `index.total` is
    // the only place the number ever appears.
    const view = reduceAll([
      { type: "status", data: { run: run("running") } },
      progress({ stage: "scan", total: 1 }),
      progress({ stage: "index", total: 10_001 }),
    ]);
    expect(view.messagesIndexed).toBe(10_001);
    expect(view.messagesParsed).toBeNull();
  });

  it("keeps the last sampled document counts when a tick reports nothing", () => {
    const view = reduceAll([
      progress({ stage: "process", docs_done: 4, docs_total: 15 }),
      progress({ stage: "process" }),
    ]);
    expect(view.docsDone).toBe(4);
    expect(view.docsTotal).toBe(15);
  });

  it("a reingest run announces its wipe", () => {
    const view = reduceAll([
      { type: "status", data: { run: run("running", { reingest: true }) } },
      progress({ stage: "scan", total: 1 }),
      progress({ stage: "reset" }),
    ]);
    expect(view.stage).toBe("reset");
    expect(graphStageLabel(view)).toBe("wiping the working directory");
  });

  it("a new running run resets a previous run's residue", () => {
    const finished = reduceAll(boundedRunEvents());
    const fresh = reduceGraphBuildEvent(finished, {
      type: "status",
      data: { run: run("running", { run_id: "g2" }) },
    });
    expect(fresh.run?.run_id).toBe("g2");
    expect(fresh.filesDone).toBe(0);
    expect(fresh.docsDone).toBeNull();
    expect(fresh.logs).toEqual([]);
  });

  it("a failed run carries its error and clears the activity line", () => {
    const view = reduceAll([
      { type: "status", data: { run: run("running") } },
      progress({ stage: "parse", file: "sms.db", current: 1, total: 1 }),
      {
        type: "status",
        data: { run: run("failed", { error: "GraphUnavailable: EACCES" }) },
      },
    ]);
    expect(view.run?.state).toBe("failed");
    expect(view.run?.error).toContain("EACCES");
    expect(view.currentFile).toBeNull();
    expect(view.stage).toBeNull();
  });

  it("caps the log tail — a multi-day backfill relays a lot", () => {
    const lines: GraphBuildEvent[] = Array.from(
      { length: MAX_GRAPH_LOG_LINES + 25 },
      (_, index) => ({
        type: "log",
        data: { level: "INFO", message: `line ${index}` },
      }),
    );
    const view = reduceAll(lines);
    expect(view.logs).toHaveLength(MAX_GRAPH_LOG_LINES);
    expect(view.logs[0].message).toBe("line 25"); // oldest dropped
  });

  it("ignores unknown progress stages (forward compatibility)", () => {
    const view = reduceGraphBuildEvent(
      initialGraphBuildView,
      progress({ stage: "quantum_dedupe" }),
    );
    expect(view).toEqual(initialGraphBuildView);
  });

  it("is pure — inputs are never mutated", () => {
    const before = reduceAll(boundedRunEvents().slice(0, 4));
    const frozen = JSON.parse(JSON.stringify(before)) as GraphBuildView;
    reduceGraphBuildEvent(before, {
      type: "log",
      data: { level: "INFO", message: "x" },
    });
    expect(before).toEqual(frozen);
  });
});

describe("graphStageLabel", () => {
  it("names each stage, and nothing when idle", () => {
    const at = (stage: string | null): string | null =>
      graphStageLabel({ ...initialGraphBuildView, stage });
    expect(at("scan")).toBe("scanning the corpus");
    expect(at("parse")).toBe("parsing");
    expect(at("bound")).toBe("applying the message bound");
    expect(at("index")).toBe("handing messages to the engine");
    expect(at("process")).toBe("extracting entities and relations");
    expect(at(null)).toBeNull();
    expect(at("quantum_dedupe")).toBeNull();
  });
});
