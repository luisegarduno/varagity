"use client";

import { PanelRightOpenIcon } from "lucide-react";
import Link from "next/link";
import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  useSyncExternalStore,
} from "react";

import { Composer } from "@/components/chat/Composer";
import { MessageList } from "@/components/chat/MessageList";
import { useChat } from "@/components/chat/use-chat";
import {
  EvidencePanel,
  EvidenceSheet,
  type EvidenceScrollTarget,
} from "@/components/provenance/EvidencePanel";
import {
  settingValue,
  useSettingsCatalog,
} from "@/components/settings/use-settings";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  evidenceRailOpen,
  evidenceRailOpenServer,
  setEvidenceRailOpen,
  subscribeDisplayPrefs,
} from "@/lib/display-prefs";
import {
  evidenceFromMessage,
  evidenceFromRetrieval,
  LIVE_EVIDENCE_KEY,
  type Evidence,
} from "@/lib/evidence";
import { onToggleEvidence } from "@/lib/ui-bus";
import { cn } from "@/lib/utils";

// Tailwind's lg — the evidence rail's breakpoint. Below it the panel
// lives in the bottom sheet instead.
const DESKTOP_QUERY = "(min-width: 1024px)";

function subscribeDesktop(onChange: () => void): () => void {
  const mql = window.matchMedia(DESKTOP_QUERY);
  mql.addEventListener("change", onChange);
  return () => mql.removeEventListener("change", onChange);
}

function isDesktopSnapshot(): boolean {
  return window.matchMedia(DESKTOP_QUERY).matches;
}

function isDesktopServerSnapshot(): boolean {
  // SSR renders the desktop shape; the rail stays CSS-hidden below lg, so
  // a phone's first paint is right before hydration flips this to false.
  return true;
}

// Any open Base UI layer that owns Escape: dialog/menu/listbox popups
// carry `data-open` while open; drawer popups unmount when closed, so
// their presence alone means open (or still animating out).
const OPEN_LAYER_SELECTOR =
  '[data-open]:is([role="dialog"], [role="alertdialog"], [role="menu"], [role="listbox"]), [data-slot=drawer-popup]';

/**
 * One conversation: transcript, streaming turn, composer — and the
 * evidence panel (spec_v2 §4.6), which follows the newest answer unless
 * the user focuses another one via its citations or sources affordance.
 * ≥lg the panel is a collapsible right rail (collapse persists via the
 * display prefs); below lg it opens as a bottom sheet.
 */
export function Conversation({ conversationId }: { conversationId: string }) {
  const { messages, turn, isStreaming, loadError, send, stop } =
    useChat(conversationId);
  // null = follow the latest answer (the live turn while one streams).
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [scrollTarget, setScrollTarget] =
    useState<EvidenceScrollTarget | null>(null);
  const [sheetOpen, setSheetOpen] = useState(false);

  const railOpen = useSyncExternalStore(
    subscribeDisplayPrefs,
    evidenceRailOpen,
    evidenceRailOpenServer,
  );
  const isDesktop = useSyncExternalStore(
    subscribeDesktop,
    isDesktopSnapshot,
    isDesktopServerSnapshot,
  );

  // Crossing up to desktop retires the sheet (the rail takes over);
  // adjust-during-render so a later resize back down doesn't resurrect it.
  const [wasDesktop, setWasDesktop] = useState(isDesktop);
  if (wasDesktop !== isDesktop) {
    setWasDesktop(isDesktop);
    if (isDesktop) setSheetOpen(false);
  }

  const { catalog } = useSettingsCatalog();
  // The stage indicator's pre-retrieval guess; the retrieval event itself
  // is the truth once it lands (lib/stage.ts).
  const rerankActive =
    settingValue(catalog, "RETRIEVAL_METHOD") === "reranked" &&
    settingValue(catalog, "RERANK_ENABLED") === true;

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

  // The composer's ↑-recall: the in-flight question, else the last sent.
  const lastUserQuery = useMemo(() => {
    if (turn) return turn.query;
    const transcript = messages ?? [];
    for (let index = transcript.length - 1; index >= 0; index -= 1) {
      if (transcript[index].role === "user") return transcript[index].content;
    }
    return null;
  }, [turn, messages]);

  /** Bring the panel into view: reopen a collapsed rail, or the sheet. */
  const revealPanel = useCallback(() => {
    if (isDesktopSnapshot()) {
      if (!evidenceRailOpen()) setEvidenceRailOpen(true);
    } else {
      setSheetOpen(true);
    }
  }, []);

  const handleCite = useCallback(
    (evidence: Evidence, chunkIndex: number) => {
      const chunk = evidence.chunks[chunkIndex];
      if (!chunk) return;
      setSelectedKey(evidence.key);
      setScrollTarget((previous) => ({
        chunkKey: chunk.key,
        nonce: (previous?.nonce ?? 0) + 1,
      }));
      revealPanel();
    },
    [revealPanel],
  );

  const handleShowEvidence = useCallback(
    (evidence: Evidence) => {
      setSelectedKey(evidence.key);
      revealPanel();
    },
    [revealPanel],
  );

  const handleSend = useCallback(
    (query: string) => {
      setSelectedKey(null); // the panel follows the new turn
      setScrollTarget(null);
      send(query);
    },
    [send],
  );

  // The error banner's way out; send() replaces the errored turn.
  const handleRetry = useCallback(() => {
    if (turn) handleSend(turn.query);
  }, [turn, handleSend]);

  const handleSheetOpenChange = useCallback((open: boolean) => {
    setSheetOpen(open);
    if (!open) setScrollTarget(null); // a reopen shouldn't replay the scroll
  }, []);

  // ⌘K's "toggle evidence" lands here: the rail ≥lg, the sheet below.
  useEffect(() => {
    return onToggleEvidence(() => {
      if (isDesktopSnapshot()) {
        setEvidenceRailOpen(!evidenceRailOpen());
      } else {
        setSheetOpen((open) => !open);
      }
    });
  }, []);

  // Esc stops the stream — unless an open layer (dialog, menu, the
  // evidence sheet) owns the key press.
  useEffect(() => {
    if (!isStreaming) return;
    function onKeyDown(event: KeyboardEvent) {
      if (event.key !== "Escape" || event.defaultPrevented) return;
      if (document.querySelector(OPEN_LAYER_SELECTOR)) return;
      stop();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [isStreaming, stop]);

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
      <div
        role="status"
        aria-busy="true"
        aria-label="Loading conversation"
        className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-6 p-6"
      >
        <Skeleton className="h-9 w-2/5 self-end rounded-xl" />
        <Skeleton className="h-24 w-4/5 rounded-xl" />
        <Skeleton className="h-9 w-1/3 self-end rounded-xl" />
        <Skeleton className="h-16 w-3/5 rounded-xl" />
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0">
      <div className="relative flex h-full min-w-0 flex-1 flex-col">
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
          rerankActive={rerankActive}
          onRetry={handleRetry}
        />
        <Composer
          onSend={handleSend}
          onStop={stop}
          isStreaming={isStreaming}
          lastUserQuery={lastUserQuery}
        />
        {isDesktop && !railOpen && (
          <Button
            variant="outline"
            size="icon-sm"
            aria-label="Show evidence panel"
            title="Show evidence panel"
            onClick={() => setEvidenceRailOpen(true)}
            className="absolute top-3 right-3 z-10 hidden text-muted-foreground shadow-xs lg:inline-flex motion-safe:animate-in motion-safe:fade-in motion-safe:zoom-in-95 motion-safe:fill-mode-backwards motion-safe:delay-200"
          >
            <PanelRightOpenIcon />
          </Button>
        )}
      </div>

      {/* Desktop rail: kept mounted (scroll + expansion survive a
          collapse), width-animated shut, inert while hidden. */}
      {isDesktop && (
        <div
          inert={!railOpen}
          className={cn(
            "relative hidden h-full shrink-0 overflow-hidden lg:block",
            "motion-safe:transition-[width] motion-safe:duration-300 motion-safe:ease-in-out",
            railOpen ? "w-96 border-l border-border" : "w-0",
          )}
        >
          <EvidencePanel
            evidence={activeEvidence}
            scrollTarget={scrollTarget}
            onClose={() => setEvidenceRailOpen(false)}
            className="absolute inset-y-0 right-0 flex w-96"
          />
        </div>
      )}

      {/* Mobile bottom sheet (the drawer renders nothing while closed). */}
      {!isDesktop && (
        <EvidenceSheet
          evidence={activeEvidence}
          scrollTarget={scrollTarget}
          open={sheetOpen}
          onOpenChange={handleSheetOpenChange}
        />
      )}
    </div>
  );
}
