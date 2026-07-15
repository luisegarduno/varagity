/**
 * The pipeline stage model behind the chat's inline stage indicator
 * (spec_v2 §4.8): "retrieving → reranking → generating", derived per
 * render from the streaming turn.
 *
 * Honest by construction: the server emits the `retrieval` SSE event only
 * *after* the rerank stage completed (spec_v2 §4.3), so the retrieve/rerank
 * boundary is not observable from the browser — the one event completes
 * both stages at once. Before it lands, whether a rerank stage is shown at
 * all comes from the settings catalog (the caller's `rerankActive` guess);
 * once it lands, `reranked_to` is the truth and overrides the guess.
 */

/** The slice of a `StreamingTurn` the stage model reads. */
export interface StageTurn {
  /** Concatenated reasoning deltas (`""` before any arrived). */
  reasoning: string;
  /** Concatenated answer deltas (`""` before any arrived). */
  answer: string;
  /** The `retrieval` event (post-rerank), `null` before it lands. */
  retrieval: { top_k: number; reranked_to: number | null } | null;
  /** The terminal `done` payload (`null` until the turn completes). */
  done: unknown;
  /** The in-band `error` payload (`null` unless the pipeline failed). */
  error: unknown;
  /** True when the user stopped the stream. */
  stopped: boolean;
}

/** One stage's lifecycle state. */
export type StageStatus = "pending" | "active" | "done" | "failed";

/** One pipeline stage, display-ready. */
export interface Stage {
  /** Stable key (React key + test hook). */
  key: "retrieve" | "rerank" | "generate";
  /** The visible label. */
  label: string;
  /** Extra mono detail, e.g. the rerank narrowing `"40 → 5"`. */
  detail: string | null;
  /** Lifecycle state. */
  status: StageStatus;
}

/**
 * Derive the ordered stage list for one streaming turn.
 *
 * Statuses: a stage is `done` once its completion signal arrived; the
 * first not-done stage is `active` while the turn is live, `failed` when
 * the turn carries an error (the failure happened *in* that stage), and
 * plain `pending` on a stopped turn (nothing is running anymore); every
 * later stage stays `pending`.
 *
 * Args:
 *   turn: The streaming turn (a `StreamingTurn` is structurally one).
 *   opts: `rerankActive` — whether the current settings put reranking on
 *     the query path; used only until the `retrieval` event reports the
 *     truth via `reranked_to`.
 */
export function deriveStages(
  turn: StageTurn,
  opts: { rerankActive: boolean },
): Stage[] {
  const rerankActive = turn.retrieval
    ? turn.retrieval.reranked_to !== null
    : opts.rerankActive;
  // A delta before the retrieval event still proves retrieval finished
  // (the protocol orders evidence before prose).
  const retrieveDone =
    turn.retrieval !== null ||
    turn.reasoning !== "" ||
    turn.answer !== "" ||
    turn.done != null;
  const generateDone = turn.done != null;
  const failed = turn.error != null;
  const settled = generateDone || failed || turn.stopped;

  const stages: Stage[] = [
    { key: "retrieve", label: "Retrieving", detail: null, status: "pending" },
  ];
  if (rerankActive) {
    stages.push({
      key: "rerank",
      label: "Reranking",
      detail:
        turn.retrieval && turn.retrieval.reranked_to !== null
          ? `${turn.retrieval.top_k} → ${turn.retrieval.reranked_to}`
          : null,
      // Completed by the same event that completes retrieve (see module doc).
      status: "pending",
    });
  }
  stages.push({
    key: "generate",
    label: "Generating",
    detail: null,
    status: "pending",
  });

  let activeAssigned = false;
  for (const stage of stages) {
    const stageDone = stage.key === "generate" ? generateDone : retrieveDone;
    if (stageDone) {
      stage.status = "done";
    } else if (!activeAssigned) {
      activeAssigned = true;
      stage.status = failed ? "failed" : settled ? "pending" : "active";
    }
  }
  return stages;
}

/** The stage currently active or failed, if any (drives announcements). */
export function currentStage(stages: readonly Stage[]): Stage | null {
  return (
    stages.find(
      (stage) => stage.status === "active" || stage.status === "failed",
    ) ?? null
  );
}
