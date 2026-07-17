/**
 * Session-scoped usage recall for completed turns.
 *
 * `messages` rows deliberately persist no token usage (the rate is a
 * property of the hardware + model at answer time, not of the answer),
 * so a turn's usage lives only as long as this JS session: recorded off
 * the `done` event when the turn folds into the transcript, gone on
 * reload. Module scope — not hook state — so revisiting a conversation
 * answered earlier in the session still shows its numbers (conversation
 * components remount per id).
 */
import type { DoneEvent } from "@/lib/api";
import { usageFromDone, type EvidenceUsage } from "@/lib/evidence";

const usageByMessageId = new Map<string, EvidenceUsage>();

/** Record a completed turn's usage under its persisted message id. */
export function recordSessionUsage(
  messageId: string,
  usage: DoneEvent["usage"],
): void {
  const normalized = usageFromDone(usage);
  if (normalized) usageByMessageId.set(messageId, normalized);
}

/** The usage recorded for a message this session, if any. */
export function sessionUsage(messageId: string): EvidenceUsage | null {
  return usageByMessageId.get(messageId) ?? null;
}

/** Test hook: drop everything recorded so far. */
export function clearSessionUsage(): void {
  usageByMessageId.clear();
}
