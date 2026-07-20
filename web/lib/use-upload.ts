"use client";

/**
 * The shared upload machinery (spec_v3 §5.3) behind both upload surfaces:
 *
 * - `useUpload` — the corpus dropzone's validate → upload → report flow.
 *   Per-file outcome rows, no ingest. Dropped folders are walked
 *   (`filesFromDrop`) and then share the composer's filter, so they upload
 *   with their structure and summarize rather than enumerate.
 * - `useComposerAttach` — the composer's 📎 flow: filter (summarized, never
 *   enumerated) → upload with relative paths → auto-ingest
 *   (`reingest: false` — it never clears the stale-corpus flag) → live
 *   progress chip over the ingest-status SSE, with a client-side queue on
 *   `409 ingest_already_running`: files are already safely on disk at that
 *   point (upload and ingest are decoupled server-side), so the attach
 *   holds and re-issues when the in-flight run's stream reaches terminal.
 *
 * The driver (`createAttachController`) takes its effects injected so the
 * state machine is unit-testable without a DOM or a network.
 */

import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useRef, useState } from "react";

import { useMountEffect } from "@/hooks/use-mount-effect";
import {
  ApiError,
  startIngest,
  streamIngestStatus,
  uploadDocuments,
  type ConfigResponse,
  type IngestEvent,
  type UploadResponse,
  type UploadedFile,
} from "@/lib/api";
import { filesFromDrop } from "@/lib/dropped-files";
import {
  initialIngestView,
  reduceIngestEvent,
  type IngestView,
} from "@/lib/ingest-reducer";
import { queryKeys } from "@/lib/queries";
import {
  countSkip,
  mergeSkipCounts,
  planAttachments,
  skipLabel,
  summarizeSkipped,
  validateUpload,
  type SkipCounts,
} from "@/lib/upload";

/** One rendered upload outcome row (the dropzone's list). */
export interface UploadOutcome {
  fileName: string;
  ok: boolean;
  detail: string;
}

/** Human phrasing for the per-file rejection reasons (client and server). */
export const REJECTION_LABELS: Record<string, string> = {
  extension_not_allowed: "file type not allowed",
  file_too_large: "over the size limit",
  invalid_filename: "invalid file name",
  invalid_path: "invalid path",
  path_too_deep: "folder nesting too deep",
};

function failureMessage(failure: unknown): string {
  return failure instanceof ApiError ? failure.message : String(failure);
}

/** Map a stored/rejected server entry onto its outcome row. */
function outcomeFromEntry(entry: UploadedFile): UploadOutcome {
  return {
    fileName: entry.file_name,
    ok: entry.stored,
    detail: entry.stored
      ? entry.replaced
        ? "replaced — re-ingest to pick up the new content"
        : "uploaded — not yet ingested"
      : ((entry.reason && REJECTION_LABELS[entry.reason]) ?? entry.reason ?? "rejected"),
  };
}

/** What `useUpload` hands the dropzone. */
export interface UploadHandle {
  busy: boolean;
  /** Per-file rows — a flat pick's files are each worth naming. */
  outcomes: UploadOutcome[];
  /** A folder drop's one-line roll-up; `null` for flat picks. */
  summary: string | null;
  /** The click-to-pick input's files (always flat — no `webkitdirectory`). */
  handleFiles: (files: File[]) => Promise<void>;
  /** A drop, descending into any folders in it. */
  handleDrop: (transfer: DataTransfer) => Promise<void>;
}

/**
 * The one line a folder drop reports instead of hundreds of rows: what
 * landed, plus the skip summary when anything was filtered.
 */
function folderSummary(stored: number, skipped: SkipCounts): string {
  const head =
    stored === 0
      ? "Nothing uploaded"
      : `${stored} ${stored === 1 ? "file" : "files"} uploaded — not yet ingested`;
  const skips = summarizeSkipped(skipped);
  return skips === null ? head : `${head} · ${skips}`;
}

/**
 * The dropzone's validate → upload → report machinery (no ingest — that's
 * the explicit action). Client-side validation mirrors the server's
 * extension/size rules (from `GET /api/config`); the server's per-file
 * outcomes render afterwards.
 *
 * Folders report differently from files, not incidentally: a picked folder
 * ignores `accept` and arrives with `.DS_Store`, `.git/` and every image in
 * it, so it's filtered and summarized (spec_v3 §5.3) rather than turned
 * into one rejection row per file.
 */
export function useUpload(
  config: ConfigResponse | null,
  onUploaded: () => void,
): UploadHandle {
  const [busy, setBusy] = useState(false);
  const [outcomes, setOutcomes] = useState<UploadOutcome[]>([]);
  const [summary, setSummary] = useState<string | null>(null);

  async function runUpload(
    files: File[],
    folder: boolean,
    walkSkips: SkipCounts,
  ): Promise<void> {
    if (busy) return;
    const skips: SkipCounts = { ...walkSkips };
    const rejected: UploadOutcome[] = [];
    let accepted: File[];
    let paths: readonly string[] | null = null;

    if (folder) {
      const plan = planAttachments(
        files,
        config?.allowed_extensions ?? null,
        config?.upload_max_mb ?? null,
        { folder: true },
      );
      accepted = plan.accepted;
      paths = plan.paths;
      mergeSkipCounts(skips, plan.skipped);
    } else {
      if (files.length === 0) return;
      accepted = [];
      for (const file of files) {
        const check = config
          ? validateUpload(
              file.name,
              file.size,
              config.allowed_extensions,
              config.upload_max_mb,
            )
          : { fileName: file.name, ok: true as const };
        if (check.ok) accepted.push(file);
        else
          rejected.push({
            fileName: check.fileName,
            ok: false,
            detail:
              (check.reason && REJECTION_LABELS[check.reason]) ??
              check.reason ??
              "rejected",
          });
      }
    }

    let stored: UploadOutcome[] = [];
    let storedCount = 0;
    let failed: string | null = null;
    if (accepted.length > 0) {
      setBusy(true);
      try {
        const response = await uploadDocuments(accepted, paths);
        storedCount = response.files.filter((entry) => entry.stored).length;
        if (folder) {
          for (const entry of response.files) {
            if (!entry.stored) countSkip(skips, skipLabel(entry.reason ?? "rejected"));
          }
        } else {
          stored = response.files.map(outcomeFromEntry);
        }
        if (storedCount > 0) onUploaded();
      } catch (failure) {
        failed = failureMessage(failure);
        stored = [{ fileName: `${accepted.length} file(s)`, ok: false, detail: failed }];
      } finally {
        setBusy(false);
      }
    }

    if (folder) {
      setOutcomes([]);
      setSummary(failed ?? folderSummary(storedCount, skips));
    } else {
      setSummary(null);
      setOutcomes([...stored, ...rejected]);
    }
  }

  return {
    busy,
    outcomes,
    summary,
    handleFiles: (files) => runUpload(files, false, {}),
    // `filesFromDrop` takes its handles before its first await, so kicking
    // it off from the handler (rather than awaiting the drop event) is safe.
    handleDrop: async (transfer) => {
      const dropped = await filesFromDrop(transfer);
      await runUpload(dropped.files, dropped.folder, dropped.skipped);
    },
  };
}

// ── The composer attach flow ────────────────────────────────────────────

export type AttachPhase =
  | "idle"
  | "uploading"
  | "queued"
  | "ingesting"
  | "done"
  | "error";

export interface AttachState {
  phase: AttachPhase;
  /** Files currently uploading (drives the "Uploading N files…" label). */
  pending: number;
  /** Files stored by this attach cycle (the "N documents added" label). */
  stored: number;
  /** One-line skip summary (client filter + server rejections), if any. */
  skipped: string | null;
  /** Live ingest view while `phase === "ingesting"`. */
  ingest: IngestView | null;
  error: string | null;
}

export const initialAttachState: AttachState = {
  phase: "idle",
  pending: 0,
  stored: 0,
  skipped: null,
  ingest: null,
  error: null,
};

const STAGE_LABELS: Record<string, string> = {
  parse: "parsing",
  chunk: "chunking",
  contextualize: "contextualizing",
  embed: "embedding",
  store: "storing",
};

/** The chip's one-line text for a state (`null` ⇒ render nothing). */
export function attachChipLabel(state: AttachState): string | null {
  switch (state.phase) {
    case "idle":
      return null;
    case "uploading":
      return `Uploading ${state.pending} ${state.pending === 1 ? "file" : "files"}…`;
    case "queued":
      return "Queued — waiting for the current ingest…";
    case "ingesting": {
      const view = state.ingest;
      const stage = view?.currentStage
        ? (STAGE_LABELS[view.currentStage] ?? view.currentStage)
        : null;
      if (stage === null) return "Ingesting…";
      const files =
        view?.filesTotal != null ? ` · ${view.filesDone}/${view.filesTotal}` : "";
      const ticks = view?.contextualize
        ? ` (${view.contextualize.done}/${view.contextualize.total})`
        : "";
      return `${stage}${files}${ticks}`;
    }
    case "done": {
      if (state.stored === 0) return state.skipped ?? "Nothing to add";
      return `${state.stored} ${state.stored === 1 ? "document" : "documents"} added`;
    }
    case "error":
      return state.error ?? "Upload failed";
  }
}

/** The attach driver's injected effects (fakes in the unit suite). */
export interface AttachDeps {
  upload: (
    files: File[],
    paths: readonly string[] | null,
  ) => Promise<UploadResponse>;
  /** `POST /api/ingest {reingest: false}` — 409s while one is in flight. */
  start: () => Promise<unknown>;
  /** The ingest-status SSE; ends at the watched run's terminal frame. */
  stream: (signal?: AbortSignal) => AsyncGenerator<IngestEvent, void, undefined>;
  onState: (state: AttachState) => void;
  /** Called after each ingest cycle settles (cache invalidation). */
  onSettled?: () => void;
}

export interface AttachOptions {
  folder: boolean;
  config: ConfigResponse | null;
}

export interface AttachController {
  attach: (files: File[], options: AttachOptions) => Promise<void>;
  /** Clear a terminal (done/error) chip back to idle. */
  dismiss: () => void;
  /** Stop streams and refuse further work (unmount). */
  abort: () => void;
}

/**
 * The composer attach state machine. One "cycle" spans everything the chip
 * currently narrates: uploads, the (possibly queued) ingest runs they
 * trigger, and the terminal summary. Attaches landing mid-cycle merge into
 * it — their files upload immediately (decoupled server-side) and the pump
 * issues one more ingest once the current run ends, since a run's
 * discovery snapshot may predate them.
 */
export function createAttachController(deps: AttachDeps): AttachController {
  let state = initialAttachState;
  let cycleActive = false;
  let needIngest = false;
  let pumping = false;
  let skips: SkipCounts = {};
  const aborter = new AbortController();

  function emit(next: Partial<AttachState>): void {
    state = { ...state, ...next };
    deps.onState(state);
  }

  async function drainStream(
    onView: ((view: IngestView) => void) | null,
  ): Promise<IngestView> {
    let view = initialIngestView;
    try {
      for await (const event of deps.stream(aborter.signal)) {
        view = reduceIngestEvent(view, event);
        onView?.(view);
      }
    } catch {
      // Stream dropped (abort, API restart) — the caller reads the last view.
    }
    return view;
  }

  async function ingestOnce(): Promise<void> {
    for (;;) {
      if (aborter.signal.aborted) return;
      try {
        await deps.start();
        break;
      } catch (failure) {
        if (failure instanceof ApiError && failure.status === 409) {
          // The runner is one-at-a-time by design and our files are already
          // safely on disk — hold, watch the in-flight run's stream to its
          // terminal frame, then re-issue. 409 stays the server's honest
          // answer; the queue is client state.
          emit({ phase: "queued" });
          await drainStream(null);
          continue;
        }
        emit({ phase: "error", error: failureMessage(failure) });
        return;
      }
    }
    emit({ phase: "ingesting", ingest: initialIngestView });
    const view = await drainStream((next) =>
      emit({ phase: "ingesting", ingest: next }),
    );
    if (aborter.signal.aborted) return;
    if (view.run?.state === "failed") {
      emit({ phase: "error", error: view.run.error ?? "The ingest failed." });
    } else {
      emit({ phase: "done", ingest: view });
    }
    deps.onSettled?.();
  }

  async function pump(): Promise<void> {
    if (pumping) return;
    pumping = true;
    try {
      while (needIngest && !aborter.signal.aborted) {
        needIngest = false;
        await ingestOnce();
      }
    } finally {
      pumping = false;
      cycleActive = false;
    }
  }

  async function attach(files: File[], options: AttachOptions): Promise<void> {
    if (files.length === 0 || aborter.signal.aborted) return;
    const fresh = !cycleActive;
    if (fresh) {
      // A new chip cycle: drop the previous one's tallies.
      cycleActive = true;
      skips = {};
      state = initialAttachState;
    }
    const plan = planAttachments(
      files,
      options.config?.allowed_extensions ?? null,
      options.config?.upload_max_mb ?? null,
      { folder: options.folder },
    );
    mergeSkipCounts(skips, plan.skipped);
    if (plan.accepted.length === 0) {
      // Nothing worth sending — the summary is the whole story, and no
      // request goes out.
      if (fresh) cycleActive = false;
      emit({
        phase: fresh ? "done" : state.phase,
        skipped: summarizeSkipped(skips),
      });
      return;
    }
    if (fresh) {
      emit({
        phase: "uploading",
        pending: plan.accepted.length,
        skipped: summarizeSkipped(skips),
        error: null,
      });
    } else {
      // A running cycle owns the phase — a merged attach only bumps counts.
      emit({
        pending: state.pending + plan.accepted.length,
        skipped: summarizeSkipped(skips),
      });
    }

    let response: UploadResponse;
    try {
      response = await deps.upload(plan.accepted, plan.paths);
    } catch (failure) {
      if (fresh && !pumping) cycleActive = false;
      emit({
        phase: fresh && !pumping ? "error" : state.phase,
        pending: Math.max(0, state.pending - plan.accepted.length),
        error: failureMessage(failure),
      });
      return;
    }
    const stored = response.files.filter((entry) => entry.stored).length;
    for (const entry of response.files) {
      if (!entry.stored) countSkip(skips, skipLabel(entry.reason ?? "rejected"));
    }
    emit({
      pending: Math.max(0, state.pending - plan.accepted.length),
      stored: state.stored + stored,
      skipped: summarizeSkipped(skips),
    });
    if (stored === 0) {
      if (fresh && !pumping) {
        cycleActive = false;
        emit({ phase: "done" });
      }
      return;
    }
    needIngest = true;
    await pump();
  }

  function dismiss(): void {
    if (state.phase === "done" || state.phase === "error") {
      state = initialAttachState;
      deps.onState(state);
    }
  }

  return { attach, dismiss, abort: () => aborter.abort() };
}

/** What `useComposerAttach` hands the composer. */
export interface ComposerAttachHandle {
  state: AttachState;
  attach: (files: File[], options: AttachOptions) => void;
  dismiss: () => void;
}

/** The composer's 📎 flow, wired to the real API and the query cache. */
export function useComposerAttach(): ComposerAttachHandle {
  const queryClient = useQueryClient();
  const [state, setState] = useState<AttachState>(initialAttachState);
  const controllerRef = useRef<AttachController | null>(null);
  if (controllerRef.current === null) {
    controllerRef.current = createAttachController({
      upload: uploadDocuments,
      start: () => startIngest(false),
      stream: (signal) => streamIngestStatus(signal),
      onState: setState,
      // A completed run changed the document table (and only that —
      // reingest:false never clears the stale-corpus flag).
      onSettled: () =>
        void queryClient.invalidateQueries({ queryKey: queryKeys.documents }),
    });
  }
  const controller = controllerRef.current;

  // The status stream outlives a render; leaving the conversation must
  // stop it. A mount-scoped cleanup over the stable controller handle.
  useMountEffect(() => () => controller.abort());

  const attach = useCallback(
    (files: File[], options: AttachOptions) => {
      void controller.attach(files, options);
    },
    [controller],
  );
  const dismiss = useCallback(() => controller.dismiss(), [controller]);
  return { state, attach, dismiss };
}
