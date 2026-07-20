"use client";

import { useQuery } from "@tanstack/react-query";
import {
  CheckIcon,
  DatabaseZapIcon,
  FileUpIcon,
  LayersIcon,
  MessageCircleQuestionIcon,
  RotateCcwIcon,
  SearchXIcon,
  TriangleAlertIcon,
} from "lucide-react";
import Link from "next/link";
import { useMemo } from "react";

import { citationComponents } from "@/components/chat/Citations";
import { Markdown } from "@/components/chat/Markdown";
import { ReasoningTrace } from "@/components/chat/ReasoningTrace";
import { Bubble, BubbleContent } from "@/components/ui/bubble";
import { Button, buttonVariants } from "@/components/ui/button";
import { Message, MessageContent } from "@/components/ui/message";
import {
  MessageScroller,
  MessageScrollerButton,
  MessageScrollerContent,
  MessageScrollerItem,
  MessageScrollerProvider,
  MessageScrollerViewport,
} from "@/components/ui/message-scroller";
import { useDebouncedValue } from "@/hooks/use-debounced-value";
import type { ChatErrorEvent, ChatMessage } from "@/lib/api";
import type { StreamingTurn } from "@/lib/chat-reducer";
import { annotateCitations } from "@/lib/citations";
import { describeChatError } from "@/lib/errors";
import type { Evidence } from "@/lib/evidence";
import { documentsQuery } from "@/lib/queries";
import { currentStage, deriveStages } from "@/lib/stage";
import { cn } from "@/lib/utils";

// New (streaming) items rise in; transcript loads render at rest.
const ENTER_ANIMATION =
  "motion-safe:animate-in motion-safe:fade-in-0 motion-safe:slide-in-from-bottom-1 motion-safe:duration-300";

function UserBubble({ text }: { text: string }) {
  return (
    <Message align="end">
      <MessageContent>
        <Bubble align="end" variant="secondary">
          <BubbleContent className="whitespace-pre-wrap border-border/60">
            {text}
          </BubbleContent>
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
 * The inline "retrieving → reranking → generating" pipeline mirror
 * (spec_v2 §4.8): done stages get a check, the active one shimmers,
 * pending ones sit dimmed, a failure marks where the pipeline died. The
 * visible row is decorative to screen readers — a polite live region
 * announces the current stage instead.
 */
function StageIndicator({
  turn,
  rerankActive,
}: {
  turn: StreamingTurn;
  rerankActive: boolean;
}) {
  const stages = deriveStages(turn, { rerankActive });
  const current = currentStage(stages);
  return (
    <div className="text-xs text-muted-foreground">
      <span
        aria-hidden="true"
        className="flex flex-wrap items-center gap-x-2 gap-y-1"
      >
        {stages.map((stage, index) => (
          <span key={stage.key} className="inline-flex items-center gap-x-2">
            {index > 0 && <span className="text-muted-foreground/40">·</span>}
            <span
              className={cn(
                "inline-flex items-center gap-1",
                stage.status === "pending" && "opacity-45",
                stage.status === "active" && "shimmer",
                stage.status === "failed" && "text-destructive",
              )}
            >
              {stage.status === "done" && <CheckIcon className="size-3" />}
              {stage.status === "failed" && (
                <TriangleAlertIcon className="size-3" />
              )}
              {stage.label}
              {stage.detail && (
                <span className="font-mono text-[11px] tabular-nums">
                  {stage.detail}
                </span>
              )}
            </span>
          </span>
        ))}
      </span>
      <span aria-live="polite" className="sr-only">
        {current &&
          (current.status === "failed"
            ? `${current.label} failed`
            : `${current.label}…`)}
      </span>
    </div>
  );
}

/** The actionable failure banner (spec_v2 §4.8): calm, with a way out. */
function ErrorBanner({
  error,
  onRetry,
}: {
  error: ChatErrorEvent;
  onRetry: () => void;
}) {
  const descriptor = describeChatError(error);
  return (
    <div
      role="alert"
      className="mt-3 w-fit max-w-full space-y-1.5 rounded-lg border border-destructive/25 bg-destructive/5 p-3 text-xs"
    >
      <p className="text-sm font-medium text-destructive">{descriptor.title}</p>
      {descriptor.hint && (
        <p className="text-muted-foreground">{descriptor.hint}</p>
      )}
      {descriptor.command && (
        <code className="block w-fit rounded-md border border-border/60 bg-muted px-1.5 py-0.5 font-mono">
          {descriptor.command}
        </code>
      )}
      {descriptor.raw && (
        <p className="font-mono break-words text-muted-foreground">
          {error.code}: {error.message}
        </p>
      )}
      <div className="flex items-center gap-3 pt-1">
        {descriptor.action === "retry" && (
          <Button size="xs" variant="outline" onClick={onRetry}>
            <RotateCcwIcon aria-hidden />
            Try again
          </Button>
        )}
        {descriptor.action === "corpus" && (
          <Link
            href="/corpus"
            className="text-muted-foreground underline underline-offset-2 hover:text-foreground"
          >
            Open the corpus
          </Link>
        )}
      </div>
    </div>
  );
}

/** Retrieval came back empty: nudge toward the corpus, quietly. */
function NoMatchesNotice() {
  return (
    <div className="mb-2 flex w-fit items-start gap-2 rounded-lg border border-border bg-muted/40 px-3 py-2 text-xs text-muted-foreground">
      <SearchXIcon className="mt-px size-3.5 shrink-0" aria-hidden />
      <p>
        Nothing matched — is the{" "}
        <Link
          href="/corpus"
          className="underline underline-offset-2 hover:text-foreground"
        >
          corpus
        </Link>{" "}
        ingested?
      </p>
    </div>
  );
}

/**
 * The pre-first-question hero. Probes the corpus once: when it turns out
 * empty, the hero becomes the guided upload → ingest → ask card
 * (spec_v2 §4.8); when the probe fails (API down), the plain hero stays —
 * the send path owns error reporting.
 */
function EmptyConversation() {
  // `undefined` while the probe is pending, and after it fails (the send
  // path owns error reporting) — both mean "unknown", i.e. the plain hero.
  const { data: documents } = useQuery(documentsQuery());
  const corpusEmpty = documents?.length === 0;

  if (corpusEmpty) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-5 p-8 text-center motion-safe:animate-in motion-safe:fade-in motion-safe:duration-300">
        <h1 className="font-heading text-3xl font-normal">
          Your corpus is empty
        </h1>
        <ol className="flex flex-wrap items-center justify-center gap-x-2 gap-y-1 text-sm text-muted-foreground">
          <li className="inline-flex items-center gap-1.5">
            <FileUpIcon className="size-3.5" aria-hidden />
            Upload documents
          </li>
          <li className="inline-flex items-center gap-1.5">
            <span aria-hidden>→</span>
            <DatabaseZapIcon className="size-3.5" aria-hidden />
            Ingest them
          </li>
          <li className="inline-flex items-center gap-1.5">
            <span aria-hidden>→</span>
            <MessageCircleQuestionIcon className="size-3.5" aria-hidden />
            Ask questions
          </li>
        </ol>
        <Link href="/corpus" className={cn(buttonVariants({ size: "sm" }))}>
          Open the corpus
        </Link>
      </div>
    );
  }

  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-3 p-8 text-center motion-safe:animate-in motion-safe:fade-in motion-safe:duration-300">
      <h1 className="font-heading text-3xl font-normal">Ask your corpus</h1>
      <p className="max-w-sm text-sm text-balance text-muted-foreground">
        Ask a question about your corpus — answers come back grounded in
        retrieved evidence, with inline citations you can inspect.
      </p>
    </div>
  );
}

/**
 * One answer's markdown with its `[SOURCE]` markers rendered as citation
 * chips, plus the "N sources" affordance that points the
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
        <div className="mt-2.5">
          <button
            type="button"
            title="Show how this answer was built"
            onClick={() => onShowEvidence(evidence)}
            className={cn(
              "inline-flex h-6 items-center gap-1.5 rounded-md border px-2 font-mono text-[11px] tabular-nums transition-colors",
              "focus-visible:ring-2 focus-visible:ring-ring/50 focus-visible:outline-none",
              active
                ? "border-primary/15 bg-primary/10 text-primary dark:border-primary/25 dark:bg-primary/15 dark:text-[oklch(0.78_calc(var(--accent-chroma)*0.7)_var(--accent-hue))]"
                : "border-border/60 bg-muted/40 text-muted-foreground hover:bg-muted hover:text-foreground",
            )}
          >
            <LayersIcon className="size-3" aria-hidden />
            {evidence.chunks.length} source
            {evidence.chunks.length === 1 ? "" : "s"}
          </button>
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
  rerankActive,
  onRetry,
}: {
  turn: StreamingTurn;
  handlers: EvidenceHandlers;
  rerankActive: boolean;
  onRetry: () => void;
}) {
  // Re-parsing partial markdown per token flashes half-styled blocks; a
  // short debounce keeps the stream feeling live without the churn.
  const debouncedAnswer = useDebouncedValue(turn.answer, 80);
  const settled = Boolean(turn.done || turn.error || turn.stopped);
  // The reasoning phase is "streaming" (auto-open) until answer tokens
  // take over or the turn settles.
  const reasoningLive = !turn.answer && !settled;
  const zeroMatches = turn.retrieval !== null && turn.retrieval.chunks.length === 0;

  return (
    <>
      <MessageScrollerItem className={ENTER_ANIMATION}>
        <UserBubble text={turn.query} />
      </MessageScrollerItem>
      <MessageScrollerItem scrollAnchor className={ENTER_ANIMATION}>
        <AssistantBubble>
          {/* On done/stop the indicator yields to the evidence panel's
              numbers; on error it stays, marking where the pipeline died. */}
          {!turn.done && !turn.stopped && (
            <div className={turn.reasoning || turn.answer ? "mb-2" : undefined}>
              <StageIndicator turn={turn} rerankActive={rerankActive} />
            </div>
          )}
          {turn.reasoning && (
            <ReasoningTrace reasoning={turn.reasoning} streaming={reasoningLive} />
          )}
          {zeroMatches && <NoMatchesNotice />}
          {debouncedAnswer && (
            <AssistantAnswer
              text={debouncedAnswer}
              evidence={handlers.liveEvidence}
              active={handlers.liveEvidence?.key === handlers.activeEvidenceKey}
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
          {turn.error && <ErrorBanner error={turn.error} onRetry={onRetry} />}
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
  rerankActive,
  onRetry,
}: {
  messages: ChatMessage[];
  turn: StreamingTurn | null;
  handlers: EvidenceHandlers;
  /** Whether current settings put reranking on the path (stage indicator). */
  rerankActive: boolean;
  /** Re-send the failed turn's question (the error banner's way out). */
  onRetry: () => void;
}) {
  const empty = messages.length === 0 && !turn;

  return (
    <MessageScrollerProvider>
      <MessageScroller className="flex-1">
        <MessageScrollerViewport>
          <MessageScrollerContent className="mx-auto w-full max-w-3xl px-4 py-6">
            {empty ? (
              <EmptyConversation />
            ) : (
              <>
                {/* The hero's h1 leaves with it — keep exactly one h1 per
                    chat page for the document outline. */}
                <h1 className="sr-only">Conversation</h1>
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
                {turn && (
                  <StreamingMessages
                    turn={turn}
                    handlers={handlers}
                    rerankActive={rerankActive}
                    onRetry={onRetry}
                  />
                )}
              </>
            )}
          </MessageScrollerContent>
        </MessageScrollerViewport>
        <MessageScrollerButton />
      </MessageScroller>
    </MessageScrollerProvider>
  );
}
