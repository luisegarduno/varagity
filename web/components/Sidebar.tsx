"use client";

import { PlusIcon, Trash2Icon } from "lucide-react";
import { usePathname, useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/ThemeToggle";
import {
  createConversation,
  deleteConversation,
  listConversations,
  type ConversationSummary,
} from "@/lib/api";
import {
  notifyConversationsChanged,
  onConversationsChanged,
} from "@/lib/conversations-bus";
import { cn } from "@/lib/utils";

/**
 * Conversation list + "new chat" + theme toggle — the app shell's left rail.
 *
 * Refetches on route changes and on the conversations bus; a trailing
 * refetch catches the background auto-title that lands a few seconds after
 * a turn persists.
 */
export function Sidebar() {
  const router = useRouter();
  const pathname = usePathname();
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [unreachable, setUnreachable] = useState(false);

  // State lands only in the promise callbacks (never synchronously in an
  // effect body) — the fetch resolution is the "external system" signal.
  const refresh = useCallback(() => {
    listConversations().then(
      (list) => {
        setConversations(list);
        setUnreachable(false);
      },
      () => setUnreachable(true),
    );
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh, pathname]);

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    const unsubscribe = onConversationsChanged(() => {
      refresh();
      clearTimeout(timer);
      timer = setTimeout(refresh, 4000); // catch the async auto-title
    });
    return () => {
      unsubscribe();
      clearTimeout(timer);
    };
  }, [refresh]);

  async function handleNewChat() {
    try {
      const created = await createConversation();
      notifyConversationsChanged();
      router.push(`/c/${created.conversation_id}`);
    } catch {
      setUnreachable(true);
    }
  }

  async function handleDelete(conversation: ConversationSummary) {
    if (!window.confirm(`Delete "${conversation.title}"?`)) return;
    await deleteConversation(conversation.conversation_id);
    notifyConversationsChanged();
    if (pathname === `/c/${conversation.conversation_id}`) {
      router.push("/");
    } else {
      void refresh();
    }
  }

  return (
    <aside className="flex h-full w-64 shrink-0 flex-col border-r border-border bg-sidebar text-sidebar-foreground">
      <div className="flex items-center justify-between gap-2 p-3">
        <span className="px-1 text-sm font-semibold tracking-tight">
          Varagity
        </span>
        <Button variant="outline" size="sm" onClick={handleNewChat}>
          <PlusIcon className="size-4" />
          New chat
        </Button>
      </div>

      <nav className="min-h-0 flex-1 overflow-y-auto px-2 pb-2">
        {unreachable ? (
          <p className="px-2 py-4 text-xs text-muted-foreground">
            API unreachable — is the stack up? (docker compose up -d)
          </p>
        ) : conversations.length === 0 ? (
          <p className="px-2 py-4 text-xs text-muted-foreground">
            No conversations yet.
          </p>
        ) : (
          <ul className="flex flex-col gap-0.5">
            {conversations.map((conversation) => {
              const href = `/c/${conversation.conversation_id}`;
              const active = pathname === href;
              return (
                <li key={conversation.conversation_id} className="group relative">
                  <button
                    type="button"
                    onClick={() => router.push(href)}
                    className={cn(
                      "w-full truncate rounded-md px-2 py-1.5 pr-8 text-left text-sm transition-colors",
                      active
                        ? "bg-sidebar-accent text-sidebar-accent-foreground"
                        : "text-sidebar-foreground/80 hover:bg-sidebar-accent/60",
                    )}
                    title={conversation.title}
                  >
                    {conversation.title}
                  </button>
                  <button
                    type="button"
                    aria-label={`Delete ${conversation.title}`}
                    onClick={() => void handleDelete(conversation)}
                    className="absolute top-1/2 right-1.5 -translate-y-1/2 rounded p-1 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100 hover:text-destructive focus-visible:opacity-100"
                  >
                    <Trash2Icon className="size-3.5" />
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </nav>

      <div className="flex items-center justify-between border-t border-border p-2">
        <ThemeToggle />
        <span className="pr-1 text-[10px] text-muted-foreground">
          local · single-user
        </span>
      </div>
    </aside>
  );
}
