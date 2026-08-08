/**
 * Which corpus a chat turn targets (spec_graphrag §4.2).
 *
 * The composer's source selector fills `ChatRequest.corpus`; a query
 * router will later fill the same field, changing no contract. Its value
 * is **derived, never persisted** (stage-2 decision #20): a conversation
 * that last answered from the graph keeps answering from the graph, and
 * the transcript already in the query cache is where that fact lives — so
 * there is no third place for it to go stale.
 */
import type { ChatMessage, TargetCorpus } from "@/lib/api";

export type { TargetCorpus };

/** The corpus a turn falls back to (also the API's own default). */
export const DEFAULT_CORPUS: TargetCorpus = "rag";

/** Selector labels — short, because they sit inside the composer row. */
export const CORPUS_LABELS: Record<TargetCorpus, string> = {
  rag: "Documents",
  graph: "Messages",
};

/** What each option grounds on, for the selector's tooltips. */
export const CORPUS_HINTS: Record<TargetCorpus, string> = {
  rag: "Answer from the document corpus (chunk retrieval).",
  graph: "Answer from the message archive's knowledge graph.",
};

/** The two options, in selector order. */
export const CORPUS_OPTIONS: readonly TargetCorpus[] = ["rag", "graph"];

/** Narrow an arbitrary persisted string to a known corpus, else `null`. */
export function asTargetCorpus(value: string | null | undefined): TargetCorpus | null {
  return value === "rag" || value === "graph" ? value : null;
}

/**
 * The corpus the newest assistant turn asked for, or `null`.
 *
 * Reads the *request*, not the outcome: a graph turn that degraded to a
 * chunk answer still persisted `target_corpus="graph"` (ADR-017 — the
 * request is part of the record), and the selector should stay where the
 * user left it rather than silently flip back on a degrade. Pre-stage-2
 * history carries `null`, which reads as chunk RAG.
 */
export function lastTargetCorpus(
  messages: readonly ChatMessage[] | null | undefined,
): TargetCorpus | null {
  if (!messages) return null;
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role !== "assistant") continue;
    const corpus = asTargetCorpus(message.target_corpus);
    if (corpus !== null) return corpus;
  }
  return null;
}

/**
 * The selector's effective value: an explicit pick wins, else the
 * conversation's own history, else documents.
 *
 * Keeping the pick as `null` until the user makes one is what lets the
 * derivation stay a derivation — the transcript can arrive (or grow) after
 * the component mounted without any state to sync.
 */
export function resolveCorpus(
  chosen: TargetCorpus | null,
  messages: readonly ChatMessage[] | null | undefined,
): TargetCorpus {
  return chosen ?? lastTargetCorpus(messages) ?? DEFAULT_CORPUS;
}
