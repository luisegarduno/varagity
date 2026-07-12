"use client";

import { LayersIcon } from "lucide-react";
import { useMemo } from "react";

import { citationComponents } from "@/components/chat/Citations";
import { Markdown, useDebouncedValue } from "@/components/chat/Markdown";
import { ReasoningTrace } from "@/components/chat/ReasoningTrace";
import { Bubble, BubbleContent } from "@/components/ui/bubble";
import { Button } from "@/components/ui/button";
import { Message, MessageContent } from "@/components/ui/message";
import {
  MessageScroller,
  MessageScrollerButton,
  MessageScrollerContent,
  MessageScrollerItem,
  MessageScrollerProvider,
  MessageScrollerViewport,
} from "@/components/ui/message-scroller";
import type { ChatMessage } from "@/lib/api";
import type { StreamingTurn } from "@/lib/chat-reducer";
import { annotateCitations } from "@/lib/citations";
import type { Evidence } from "@/lib/evidence";

function UserBubble({ text }: { text: string }) {
  return (
    <Message align="end">
      <MessageContent>
        <Bubble align="end" variant="secondary">
          <BubbleContent className="whitespace-pre-wrap">{text}</BubbleContent>
        </Bubble>
      </MessageContent>
    </Message>
  );
}

function AssistantBubble({ children }: { children: React.ReactNode }) {
  return (
    <Message align="start">
      <MessageContent>
        <Bubble align="start" variant="ghost">
          <BubbleContent>{children}</BubbleContent>
        </Bubble>
      </MessageContent>
    </Message>
  );
}

/**
 * One answer's markdown with its `[SOURCE]` markers rendered as citation
 * chips (Phase 4), plus the "N sources" affordance that points the
 * evidence panel at this answer.
 */
function AssistantAnswer({
  text,
  evidence,
  active,
  streaming = false,
  onCite,
  onShowEvidence,
}: {
  text: string;
  evidence: Evidence | null;
  active: boolean;
  /** True while the answer is still streaming (softens unmatched chips). */
  streaming?: boolean;
  onCite: (evidence: Evidence, chunkIndex: number) => void;
  onShowEvidence: (evidence: Evidence) => void;
}) {
  const annotated = useMemo(
    () => annotateCitations(text, evidence?.chunks ?? []),
    [text, evidence],
  );
  // Markdown is memo()ed on prop identity — keep the override stable.
  const components = useMemo(
    () =>
      citationComponents(
        annotated.citations,
        (chunkIndex) => {
          if (evidence) onCite(evidence, chunkIndex);
        },
        streaming,
      ),
    [annotated.citations, evidence, onCite, streaming],
  );

  return (
    <>
      <Markdown text={annotated.markdown} components={components} />
      {evidence && evidence.chunks.length > 0 && (
        <div className="mt-2">
          <Button
            variant={active ? "secondary" : "ghost"}
            size="xs"
            className="text-muted-foreground"
            title="Show how this answer was built"
            onClick={() => onShowEvidence(evidence)}
          >
            <LayersIcon aria-hidden />
            {evidence.chunks.length} source
            {evidence.chunks.length === 1 ? "" : "s"}
          </Button>
        </div>
      )}
    </>
  );
}

/** What the transcript needs to drive the evidence panel. */
export interface EvidenceHandlers {
  /** Evidence per persisted assistant message id. */
  evidenceById: ReadonlyMap<string, Evidence>;
  /** Evidence of the in-flight turn (once its `retrieval` event landed). */
  liveEvidence: Evidence | null;
  /** Which evidence the panel currently shows (affordance highlight). */
  activeEvidenceKey: string | null;
  /** A citation chip was clicked: focus that evidence + chunk. */
  onCite: (evidence: Evidence, chunkIndex: number) => void;
  /** The per-answer sources affordance was clicked. */
  onShowEvidence: (evidence: Evidence) => void;
}

/** The in-flight turn: optimistic user bubble + live assistant text. */
function StreamingMessages({
  turn,
  handlers,
}: {
  turn: StreamingTurn;
  handlers: EvidenceHandlers;
}) {
  // Re-parsing partial markdown per token flashes half-styled blocks; a
  // short debounce keeps the stream feeling live without the churn.
  const debouncedAnswer = useDebouncedValue(turn.answer, 80);
  const settled = Boolean(turn.done || turn.error || turn.stopped);
  const waiting = !turn.reasoning && !turn.answer && !settled;
  // The reasoning phase is "streaming" (auto-open) until answer tokens
  // take over or the turn settles.
  const reasoningLive = !turn.answer && !settled;

  return (
    <>
      <MessageScrollerItem>
        <UserBubble text={turn.query} />
      </MessageScrollerItem>
      <MessageScrollerItem scrollAnchor>
        <AssistantBubble>
          {waiting ? (
            <p className="animate-pulse text-sm text-muted-foreground">
              Retrieving…
            </p>
          ) : (
            <>
              {turn.reasoning && (
                <ReasoningTrace
                  reasoning={turn.reasoning}
                  streaming={reasoningLive}
                />
              )}
              {debouncedAnswer && (
                <AssistantAnswer
                  text={debouncedAnswer}
                  evidence={handlers.liveEvidence}
                  active={
                    handlers.liveEvidence?.key === handlers.activeEvidenceKey
                  }
                  streaming={!settled}
                  onCite={handlers.onCite}
                  onShowEvidence={handlers.onShowEvidence}
                />
              )}
              {turn.stopped && (
                <p className="mt-2 text-xs text-muted-foreground italic">
                  Stopped — this partial answer isn&apos;t saved.
                </p>
              )}
              {turn.error && (
                <p className="mt-2 rounded-md bg-destructive/10 px-2 py-1 text-xs text-destructive">
                  {turn.error.code}: {turn.error.message}
                </p>
              )}
            </>
          )}
        </AssistantBubble>
      </MessageScrollerItem>
    </>
  );
}

/**
 * The transcript + the streaming turn, inside the autoscrolling
 * message-scroller (pinned to the end while tokens arrive; a scroll-away
 * shows the jump-to-bottom button). Each assistant answer renders its
 * citations as chips and hands its evidence to the panel via `handlers`.
 */
export function MessageList({
  messages,
  turn,
  handlers,
}: {
  messages: ChatMessage[];
  turn: StreamingTurn | null;
  handlers: EvidenceHandlers;
}) {
  const empty = messages.length === 0 && !turn;

  return (
    <MessageScrollerProvider>
      <MessageScroller className="flex-1">
        <MessageScrollerViewport>
          <MessageScrollerContent className="mx-auto w-full max-w-3xl px-4 py-6">
            {empty ? (
              <div className="flex flex-1 items-center justify-center">
                <p className="text-sm text-muted-foreground">
                  Ask a question about your corpus.
                </p>
              </div>
            ) : (
              <>
                {messages.map((message) => (
                  <MessageScrollerItem key={message.message_id}>
                    {message.role === "user" ? (
                      <UserBubble text={message.content} />
                    ) : (
                      <AssistantBubble>
                        {message.reasoning && (
                          <ReasoningTrace
                            reasoning={message.reasoning}
                            streaming={false}
                          />
                        )}
                        <AssistantAnswer
                          text={message.content}
                          evidence={
                            handlers.evidenceById.get(message.message_id) ??
                            null
                          }
                          active={
                            message.message_id === handlers.activeEvidenceKey
                          }
                          onCite={handlers.onCite}
                          onShowEvidence={handlers.onShowEvidence}
                        />
                      </AssistantBubble>
                    )}
                  </MessageScrollerItem>
                ))}
                {turn && <StreamingMessages turn={turn} handlers={handlers} />}
              </>
            )}
          </MessageScrollerContent>
        </MessageScrollerViewport>
        <MessageScrollerButton />
      </MessageScroller>
    </MessageScrollerProvider>
  );
}
