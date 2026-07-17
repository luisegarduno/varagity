"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useMemo, useRef, useState } from "react";

import { useMountEffect } from "@/hooks/use-mount-effect";
import { ApiError, streamChat, type ChatMessage } from "@/lib/api";
import {
  newTurn,
  reduceChatEvent,
  type StreamingTurn,
} from "@/lib/chat-reducer";
import { notifyConversationsChanged } from "@/lib/conversations-bus";
import { assistantMessageFromTurn } from "@/lib/evidence";
import { conversationQuery } from "@/lib/queries";
import { recordSessionUsage } from "@/lib/session-usage";

/** What `useChat` exposes to the conversation UI. */
export interface ChatState {
  /** Persisted transcript (`null` while loading). */
  messages: ChatMessage[] | null;
  /** The in-flight (or just-failed/stopped) turn, `null` when idle. */
  turn: StreamingTurn | null;
  /** True while a stream is open. */
  isStreaming: boolean;
  /** Transcript-load failure (404 → surfaced as not-found). */
  loadError: ApiError | null;
  send: (query: string) => void;
  stop: () => void;
}

/** A locally-built transcript entry for a turn the server just persisted. */
function localMessage(
  role: "user" | "assistant",
  content: string,
  id: string,
): ChatMessage {
  return {
    message_id: id,
    role,
    content,
    created_at: new Date().toISOString(),
    sources: [],
  };
}

/**
 * Own one conversation's transcript + streaming turn.
 *
 * `send` drives the SSE loop through the token-accumulation reducer; on
 * `done` the turn folds into the cached transcript exactly as the server
 * persisted it (the authoritative answer rides in the event), so no
 * refetch is needed to render what was just answered. A stopped stream
 * keeps its partial text visible but flagged — the server persists nothing
 * for aborted turns.
 */
export function useChat(conversationId: string): ChatState {
  const queryClient = useQueryClient();
  const [turn, setTurn] = useState<StreamingTurn | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  // No reset-on-id-change here: the conversation page keys its component
  // by conversation id, so switching remounts with fresh initial state.
  const { data, error } = useQuery(conversationQuery(conversationId));
  const messages = data?.messages ?? null;
  const loadError = useMemo(() => {
    if (error === null) return null;
    return error instanceof ApiError
      ? error
      : new ApiError(0, "network_error", String(error));
  }, [error]);

  // The stream outlives a render, so leaving the conversation has to stop
  // it explicitly. A ref is exactly the stable handle a mount-scoped
  // cleanup can close over.
  useMountEffect(() => () => abortRef.current?.abort());

  const send = useCallback(
    (query: string) => {
      const trimmed = query.trim();
      if (!trimmed || abortRef.current) return;

      const controller = new AbortController();
      abortRef.current = controller;
      setIsStreaming(true);
      let current = newTurn(trimmed);
      setTurn(current);

      void (async () => {
        try {
          const events = streamChat(
            { query: trimmed, conversation_id: conversationId },
            controller.signal,
          );
          for await (const event of events) {
            current = reduceChatEvent(current, event);
            setTurn(current);
          }
          if (current.done) {
            const done = current.done;
            const settled = current;
            // Session-only recall: the fold below renders evidence from
            // the persisted message shape, which carries no usage — this
            // map is where the panel finds the turn's tokens + rate until
            // the next reload.
            recordSessionUsage(done.message_id, done.usage);
            // Fold the turn into the cache as the server persisted it —
            // evidence snapshot, reasoning, latency — so the just-answered
            // turn renders exactly like a reload (the evidence panel
            // included) without a round trip. Returning `undefined` from
            // the updater is TanStack's bail-out, which is the right
            // answer if the transcript somehow isn't cached.
            queryClient.setQueryData(
              conversationQuery(conversationId).queryKey,
              (previous) =>
                previous && {
                  ...previous,
                  messages: [
                    ...previous.messages,
                    localMessage("user", trimmed, `${done.message_id}-user`),
                    assistantMessageFromTurn(
                      done,
                      settled.retrieval,
                      settled.reasoning,
                    ),
                  ],
                },
            );
            setTurn(null);
            notifyConversationsChanged(); // list order + the async auto-title
          }
        } catch (error) {
          if (controller.signal.aborted) {
            current = { ...current, stopped: true };
          } else {
            const failure =
              error instanceof ApiError
                ? { code: error.code, message: error.message }
                : { code: "network_error", message: String(error) };
            current = { ...current, error: failure };
          }
          setTurn(current);
        } finally {
          abortRef.current = null;
          setIsStreaming(false);
        }
      })();
    },
    [conversationId, queryClient],
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  return { messages, turn, isStreaming, loadError, send, stop };
}
