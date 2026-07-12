"use client";

import Link from "next/link";

import { Composer } from "@/components/chat/Composer";
import { MessageList } from "@/components/chat/MessageList";
import { useChat } from "@/components/chat/use-chat";

/** One conversation: transcript, streaming turn, composer. */
export function Conversation({ conversationId }: { conversationId: string }) {
  const { messages, turn, isStreaming, loadError, send, stop } =
    useChat(conversationId);

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
    <div className="flex h-full min-h-0 flex-col">
      <MessageList messages={messages} turn={turn} />
      <Composer onSend={send} onStop={stop} isStreaming={isStreaming} />
    </div>
  );
}
