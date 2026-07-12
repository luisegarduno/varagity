"use client";

import Link from "next/link";
import { useCallback, useMemo, useState } from "react";

import { Composer } from "@/components/chat/Composer";
import { MessageList } from "@/components/chat/MessageList";
import { useChat } from "@/components/chat/use-chat";
import {
  EvidencePanel,
  type EvidenceScrollTarget,
} from "@/components/provenance/EvidencePanel";
import {
  evidenceFromMessage,
  evidenceFromRetrieval,
  LIVE_EVIDENCE_KEY,
  type Evidence,
} from "@/lib/evidence";

/**
 * One conversation: transcript, streaming turn, composer — and the
 * evidence panel (Phase 4), which follows the newest answer unless the
 * user focuses another one via its citations or sources affordance.
 */
export function Conversation({ conversationId }: { conversationId: string }) {
  const { messages, turn, isStreaming, loadError, send, stop } =
    useChat(conversationId);
  // null = follow the latest answer (the live turn while one streams).
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [scrollTarget, setScrollTarget] =
    useState<EvidenceScrollTarget | null>(null);

  // Each assistant message's evidence, keyed by message id; the closest
  // preceding user message is the query that drives term highlighting.
  const evidenceById = useMemo(() => {
    const map = new Map<string, Evidence>();
    let lastQuery: string | null = null;
    for (const message of messages ?? []) {
      if (message.role === "user") {
        lastQuery = message.content;
        continue;
      }
      const evidence = evidenceFromMessage(message, lastQuery);
      if (evidence) map.set(message.message_id, evidence);
    }
    return map;
  }, [messages]);

  const liveEvidence = useMemo(() => {
    if (!turn?.retrieval) return null;
    return evidenceFromRetrieval(turn.retrieval, {
      query: turn.query,
      latencyMs: turn.done?.usage.latency_ms ?? null,
    });
  }, [turn]);

  const newestEvidence = useMemo(() => {
    if (liveEvidence) return liveEvidence;
    const transcript = messages ?? [];
    for (let index = transcript.length - 1; index >= 0; index -= 1) {
      const evidence = evidenceById.get(transcript[index].message_id);
      if (evidence) return evidence;
    }
    return null;
  }, [liveEvidence, messages, evidenceById]);

  const selectedEvidence =
    selectedKey === LIVE_EVIDENCE_KEY
      ? liveEvidence
      : selectedKey !== null
        ? (evidenceById.get(selectedKey) ?? null)
        : null;
  const activeEvidence = selectedEvidence ?? newestEvidence;

  const handleCite = useCallback(
    (evidence: Evidence, chunkIndex: number) => {
      const chunk = evidence.chunks[chunkIndex];
      if (!chunk) return;
      setSelectedKey(evidence.key);
      setScrollTarget((previous) => ({
        chunkKey: chunk.key,
        nonce: (previous?.nonce ?? 0) + 1,
      }));
    },
    [],
  );

  const handleShowEvidence = useCallback((evidence: Evidence) => {
    setSelectedKey(evidence.key);
  }, []);

  const handleSend = useCallback(
    (query: string) => {
      setSelectedKey(null); // the panel follows the new turn
      setScrollTarget(null);
      send(query);
    },
    [send],
  );

  if (loadError) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-2 p-8 text-center">
        <p className="text-sm font-medium">
          {loadError.status === 404
            ? "This conversation doesn't exist (anymore)."
            : `Couldn't load the conversation — ${loadError.message}`}
        </p>
        <Link href="/" className="text-sm text-muted-foreground underline">
          Back to the newest chat
        </Link>
      </div>
    );
  }

  if (messages === null) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <p className="animate-pulse text-sm text-muted-foreground">Loading…</p>
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0">
      <div className="flex h-full min-w-0 flex-1 flex-col">
        <MessageList
          messages={messages}
          turn={turn}
          handlers={{
            evidenceById,
            liveEvidence,
            activeEvidenceKey: activeEvidence?.key ?? null,
            onCite: handleCite,
            onShowEvidence: handleShowEvidence,
          }}
        />
        <Composer onSend={handleSend} onStop={stop} isStreaming={isStreaming} />
      </div>
      {/* A plain always-on rail ≥lg; the responsive bottom-sheet treatment
          is Phase 9. */}
      <EvidencePanel
        evidence={activeEvidence}
        scrollTarget={scrollTarget}
        className="hidden w-96 shrink-0 border-l border-border lg:flex"
      />
    </div>
  );
}
