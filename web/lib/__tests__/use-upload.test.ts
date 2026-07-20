import { describe, expect, it, vi } from "vitest";

import {
  ApiError,
  type ConfigResponse,
  type IngestEvent,
  type IngestRun,
  type UploadResponse,
} from "@/lib/api";
import {
  attachChipLabel,
  createAttachController,
  initialAttachState,
  REJECTION_LABELS,
  type AttachDeps,
  type AttachState,
} from "@/lib/use-upload";
import { initialIngestView } from "@/lib/ingest-reducer";

import { pickedFile } from "./helpers";

const CONFIG = {
  allowed_extensions: [".md", ".txt"],
  upload_max_mb: 1,
} as unknown as ConfigResponse;

function makeRun(id: string, state: string, error: string | null = null): IngestRun {
  return {
    run_id: id,
    state,
    reingest: false,
    started_at: "2026-07-16T00:00:00Z",
    finished_at: state === "running" ? null : "2026-07-16T00:00:05Z",
    summary: null,
    error,
  };
}

function progress(
  stage: string,
  fields: { file?: string; total?: number } = {},
): IngestEvent {
  return {
    type: "progress",
    data: {
      stage,
      file: fields.file ?? null,
      outcome: null,
      current: null,
      total: fields.total ?? null,
      files_done: null,
      files_total: null,
    },
  };
}

function runEvents(
  id: string,
  state: "completed" | "failed" = "completed",
  error: string | null = null,
): IngestEvent[] {
  return [
    { type: "status", data: { run: makeRun(id, "running") } },
    progress("discover", { total: 2 }),
    progress("parse", { file: "a.md" }),
    { type: "status", data: { run: makeRun(id, state, error) } },
  ];
}

async function* streamOf(
  events: IngestEvent[],
): AsyncGenerator<IngestEvent, void, undefined> {
  for (const event of events) yield event;
}

function uploadOk(files: File[]): UploadResponse {
  return {
    files: files.map((file) => ({
      file_name: file.name,
      size_bytes: file.size,
      stored: true,
      replaced: false,
      reason: null,
      relative_path: null,
    })),
  };
}

function deferred(): { promise: Promise<void>; resolve: () => void } {
  let resolve!: () => void;
  const promise = new Promise<void>((r) => (resolve = r));
  return { promise, resolve };
}

/** A controller wired to fakes, recording every emitted state. */
function harness(overrides: Partial<AttachDeps> = {}) {
  const states: AttachState[] = [];
  const deps: AttachDeps = {
    upload: vi.fn(async (files: File[]) => uploadOk(files)),
    start: vi.fn(async () => ({})),
    stream: vi.fn(() => streamOf(runEvents("run-1"))),
    onState: (state) => states.push(state),
    onSettled: vi.fn(),
    ...overrides,
  };
  const controller = createAttachController(deps);
  return {
    controller,
    deps,
    states,
    latest: () => states[states.length - 1],
    phases: () => states.map((state) => state.phase),
  };
}

describe("createAttachController", () => {
  it("uploads, ingests, and lands on done with the stored count", async () => {
    const { controller, deps, latest, phases } = harness();
    await controller.attach(
      [pickedFile("a.md"), pickedFile("b.md"), pickedFile("c.md")],
      { folder: false, config: CONFIG },
    );
    expect(deps.upload).toHaveBeenCalledWith(expect.any(Array), null);
    expect(phases()).toContain("uploading");
    expect(phases()).toContain("ingesting");
    expect(latest().phase).toBe("done");
    expect(latest().stored).toBe(3);
    expect(attachChipLabel(latest())).toBe("3 documents added");
    expect(deps.onSettled).toHaveBeenCalledTimes(1);
  });

  it("sends folder uploads with positionally aligned paths", async () => {
    const { controller, deps } = harness();
    await controller.attach(
      [pickedFile("a.md", "corpus/q3/a.md"), pickedFile("b.md", "corpus/q4/b.md")],
      { folder: true, config: CONFIG },
    );
    expect(deps.upload).toHaveBeenCalledWith(expect.any(Array), [
      "corpus/q3/a.md",
      "corpus/q4/b.md",
    ]);
  });

  it("queues on 409 and re-issues once the in-flight run drains", async () => {
    const start = vi
      .fn()
      .mockRejectedValueOnce(
        new ApiError(409, "ingest_already_running", "one at a time"),
      )
      .mockResolvedValue({});
    const stream = vi
      .fn()
      // First stream: the foreign in-flight run being waited out.
      .mockReturnValueOnce(streamOf(runEvents("foreign-run")))
      // Second stream: our run.
      .mockReturnValue(streamOf(runEvents("our-run")));
    const { controller, latest, phases } = harness({ start, stream });

    await controller.attach([pickedFile("a.md")], { folder: false, config: CONFIG });

    expect(phases()).toContain("queued");
    expect(start).toHaveBeenCalledTimes(2);
    expect(stream).toHaveBeenCalledTimes(2);
    expect(latest().phase).toBe("done");
    expect(latest().stored).toBe(1);
  });

  it("keeps re-queueing while the runner stays busy", async () => {
    const start = vi
      .fn()
      .mockRejectedValueOnce(new ApiError(409, "ingest_already_running", "busy"))
      .mockRejectedValueOnce(new ApiError(409, "ingest_already_running", "busy"))
      .mockResolvedValue({});
    const { controller, deps, latest } = harness({ start });
    await controller.attach([pickedFile("a.md")], { folder: false, config: CONFIG });
    expect(start).toHaveBeenCalledTimes(3);
    expect(deps.stream).toHaveBeenCalledTimes(3); // two drains + our run
    expect(latest().phase).toBe("done");
  });

  it("merges a second attach into the running cycle and ingests once more", async () => {
    const gate = deferred();
    async function* gatedRun(): AsyncGenerator<IngestEvent, void, undefined> {
      yield { type: "status", data: { run: makeRun("run-1", "running") } };
      await gate.promise;
      yield { type: "status", data: { run: makeRun("run-1", "completed") } };
    }
    const stream = vi
      .fn()
      .mockReturnValueOnce(gatedRun())
      .mockReturnValue(streamOf(runEvents("run-2")));
    const { controller, deps, latest } = harness({ stream });

    const first = controller.attach([pickedFile("a.md"), pickedFile("b.md")], {
      folder: false,
      config: CONFIG,
    });
    await vi.waitFor(() => expect(latest().phase).toBe("ingesting"));

    // Lands mid-run: uploads immediately, then rides the pump — the active
    // run's discovery snapshot may predate these files.
    await controller.attach([pickedFile("c.md")], { folder: false, config: CONFIG });
    expect(latest().stored).toBe(3);

    gate.resolve();
    await first;
    expect(deps.start).toHaveBeenCalledTimes(2);
    expect(latest().phase).toBe("done");
    expect(attachChipLabel(latest())).toBe("3 documents added");
  });

  it("sends nothing when every file is filtered, and says why", async () => {
    const { controller, deps, latest } = harness();
    await controller.attach([pickedFile("virus.exe")], {
      folder: false,
      config: CONFIG,
    });
    expect(deps.upload).not.toHaveBeenCalled();
    expect(deps.start).not.toHaveBeenCalled();
    expect(latest().phase).toBe("done");
    expect(latest().stored).toBe(0);
    expect(attachChipLabel(latest())).toBe("1 file skipped — unsupported type");
  });

  it("folds server-side rejections into the skip summary", async () => {
    const upload = vi.fn(
      async (): Promise<UploadResponse> => ({
        files: [
          {
            file_name: "kept.md",
            size_bytes: 8,
            stored: true,
            replaced: false,
            reason: null,
            relative_path: "q3/kept.md",
          },
          {
            file_name: "a/b/c/deep.md",
            size_bytes: 0,
            stored: false,
            reason: "path_too_deep",
            replaced: false,
            relative_path: null,
          },
        ],
      }),
    );
    const { controller, latest } = harness({ upload });
    await controller.attach(
      [pickedFile("kept.md", "q3/kept.md"), pickedFile("deep.md", "a/b/c/deep.md")],
      { folder: true, config: CONFIG },
    );
    expect(latest().phase).toBe("done");
    expect(latest().stored).toBe(1);
    expect(latest().skipped).toBe("1 file skipped — folder too deep");
  });

  it("lands on error when the upload itself fails", async () => {
    const upload = vi
      .fn()
      .mockRejectedValue(new ApiError(500, "docs_path_not_writable", "mount is read-only"));
    const { controller, deps, latest } = harness({ upload });
    await controller.attach([pickedFile("a.md")], { folder: false, config: CONFIG });
    expect(latest().phase).toBe("error");
    expect(attachChipLabel(latest())).toBe("mount is read-only");
    expect(deps.start).not.toHaveBeenCalled();
  });

  it("lands on error when the run fails", async () => {
    const stream = vi.fn(() => streamOf(runEvents("run-1", "failed", "boom")));
    const { controller, latest } = harness({ stream });
    await controller.attach([pickedFile("a.md")], { folder: false, config: CONFIG });
    expect(latest().phase).toBe("error");
    expect(latest().error).toBe("boom");
  });

  it("dismiss clears a terminal chip and ignores an active one", async () => {
    const { controller, latest } = harness();
    await controller.attach([pickedFile("a.md")], { folder: false, config: CONFIG });
    expect(latest().phase).toBe("done");
    controller.dismiss();
    expect(latest().phase).toBe("idle");
  });

  it("a fresh attach after a terminal chip starts a clean tally", async () => {
    const { controller, latest } = harness();
    await controller.attach([pickedFile("virus.exe")], { folder: false, config: CONFIG });
    expect(latest().skipped).toBe("1 file skipped — unsupported type");
    await controller.attach([pickedFile("a.md")], { folder: false, config: CONFIG });
    expect(latest().stored).toBe(1);
    expect(latest().skipped).toBeNull();
  });

  it("refuses further work after abort", async () => {
    const { controller, deps } = harness();
    controller.abort();
    await controller.attach([pickedFile("a.md")], { folder: false, config: CONFIG });
    expect(deps.upload).not.toHaveBeenCalled();
  });
});

describe("attachChipLabel", () => {
  const at = (partial: Partial<AttachState>): AttachState => ({
    ...initialAttachState,
    ...partial,
  });

  it("is null while idle", () => {
    expect(attachChipLabel(initialAttachState)).toBeNull();
  });

  it("counts the upload, singular and plural", () => {
    expect(attachChipLabel(at({ phase: "uploading", pending: 1 }))).toBe(
      "Uploading 1 file…",
    );
    expect(attachChipLabel(at({ phase: "uploading", pending: 3 }))).toBe(
      "Uploading 3 files…",
    );
  });

  it("names the queue wait", () => {
    expect(attachChipLabel(at({ phase: "queued" }))).toBe(
      "Queued — waiting for the current ingest…",
    );
  });

  it("narrates the ingest stage with file counts and contextualize ticks", () => {
    expect(
      attachChipLabel(
        at({
          phase: "ingesting",
          ingest: {
            ...initialIngestView,
            currentStage: "contextualize",
            filesDone: 1,
            filesTotal: 3,
            contextualize: { done: 4, total: 12 },
          },
        }),
      ),
    ).toBe("contextualizing · 1/3 (4/12)");
    expect(
      attachChipLabel(at({ phase: "ingesting", ingest: initialIngestView })),
    ).toBe("Ingesting…");
  });

  it("falls back to the skip summary when nothing was added", () => {
    expect(
      attachChipLabel(
        at({ phase: "done", stored: 0, skipped: "2 files skipped — hidden" }),
      ),
    ).toBe("2 files skipped — hidden");
    expect(attachChipLabel(at({ phase: "done", stored: 0 }))).toBe("Nothing to add");
  });

  it("shows the error text", () => {
    expect(attachChipLabel(at({ phase: "error", error: "boom" }))).toBe("boom");
    expect(attachChipLabel(at({ phase: "error" }))).toBe("Upload failed");
  });
});

describe("REJECTION_LABELS", () => {
  it("covers the v3 path reasons", () => {
    expect(REJECTION_LABELS.invalid_path).toBe("invalid path");
    expect(REJECTION_LABELS.path_too_deep).toBe("folder nesting too deep");
  });
});
