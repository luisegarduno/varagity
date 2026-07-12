"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import {
  ApiError,
  getConversation,
  streamChat,
  type ChatMessage,
} from "@/lib/api";
import {
  newTurn,
  reduceChatEvent,
  type StreamingTurn,
} from "@/lib/chat-reducer";
import { notifyConversationsChanged } from "@/lib/conversations-bus";

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
 * `done` the turn folds into the transcript exactly as the server
 * persisted it (the authoritative answer rides in the event). A stopped
 * stream keeps its partial text visible but flagged — the server persists
 * nothing for aborted turns.
 */
export function useChat(conversationId: string): ChatState {
  const [messages, setMessages] = useState<ChatMessage[] | null>(null);
  const [turn, setTurn] = useState<StreamingTurn | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [loadError, setLoadError] = useState<ApiError | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // No reset-on-id-change here: the conversation page keys its component
  // by conversation id, so switching remounts with fresh initial state.
  useEffect(() => {
    let cancelled = false;
    getConversation(conversationId).then(
      (detail) => {
        if (!cancelled) setMessages(detail.messages);
      },
      (error: unknown) => {
        if (!cancelled) {
          setLoadError(
            error instanceof ApiError
              ? error
              : new ApiError(0, "network_error", String(error)),
          );
        }
      },
    );
    return () => {
      cancelled = true;
      abortRef.current?.abort();
    };
  }, [conversationId]);

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
            setMessages((previous) => [
              ...(previous ?? []),
              localMessage("user", trimmed, `${done.message_id}-user`),
              localMessage("assistant", done.answer, done.message_id),
            ]);
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
    [conversationId],
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  return { messages, turn, isStreaming, loadError, send, stop };
}
