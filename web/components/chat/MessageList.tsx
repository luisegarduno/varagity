"use client";

import { Bubble, BubbleContent } from "@/components/ui/bubble";
import { Message, MessageContent } from "@/components/ui/message";
import {
  MessageScroller,
  MessageScrollerButton,
  MessageScrollerContent,
  MessageScrollerItem,
  MessageScrollerProvider,
  MessageScrollerViewport,
} from "@/components/ui/message-scroller";
import { Markdown, useDebouncedValue } from "@/components/chat/Markdown";
import type { ChatMessage } from "@/lib/api";
import type { StreamingTurn } from "@/lib/chat-reducer";

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

/** The in-flight turn: optimistic user bubble + live assistant text. */
function StreamingMessages({ turn }: { turn: StreamingTurn }) {
  // Re-parsing partial markdown per token flashes half-styled blocks; a
  // short debounce keeps the stream feeling live without the churn.
  const debouncedAnswer = useDebouncedValue(turn.answer, 80);
  const thinking = !turn.answer && !turn.error && !turn.done && !turn.stopped;

  return (
    <>
      <MessageScrollerItem>
        <UserBubble text={turn.query} />
      </MessageScrollerItem>
      <MessageScrollerItem scrollAnchor>
        <AssistantBubble>
          {thinking ? (
            <p className="animate-pulse text-sm text-muted-foreground">
              {turn.reasoning ? "Thinking…" : "Retrieving…"}
            </p>
          ) : (
            <>
              {debouncedAnswer && <Markdown text={debouncedAnswer} />}
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
 * shows the jump-to-bottom button).
 *
 * The `retrieval` payload each turn stashes is deliberately not rendered —
 * the evidence panel is Phase 4.
 */
export function MessageList({
  messages,
  turn,
}: {
  messages: ChatMessage[];
  turn: StreamingTurn | null;
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
                        <Markdown text={message.content} />
                      </AssistantBubble>
                    )}
                  </MessageScrollerItem>
                ))}
                {turn && <StreamingMessages turn={turn} />}
              </>
            )}
          </MessageScrollerContent>
        </MessageScrollerViewport>
        <MessageScrollerButton />
      </MessageScroller>
    </MessageScrollerProvider>
  );
}
