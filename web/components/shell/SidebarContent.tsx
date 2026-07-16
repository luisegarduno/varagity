"use client";

import { useQuery } from "@tanstack/react-query";
import { DatabaseIcon, PlusIcon, Trash2Icon } from "lucide-react";
import { usePathname, useRouter } from "next/navigation";
import { useState } from "react";

import { SettingsDrawer } from "@/components/settings/SettingsDrawer";
import { ThemeToggle } from "@/components/ThemeToggle";
import {
  AlertDialog,
  AlertDialogClose,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import {
  ApiError,
  createConversation,
  deleteConversation,
  type ConversationSummary,
} from "@/lib/api";
import { notifyConversationsChanged } from "@/lib/conversations-bus";
import { conversationsQuery } from "@/lib/queries";
import { cn } from "@/lib/utils";

/**
 * The navigation rail's inner content — wordmark, "new chat", the
 * conversation list, and the Corpus/Settings/theme footer. Rendered by the
 * persistent desktop `Sidebar` and by the mobile navigation drawer
 * (`MobileTopBar`), which passes `onNavigate` so the drawer can close
 * before a route change.
 *
 * The list is the shared `conversations` query, so mutations anywhere —
 * here, the ⌘K palette, a persisted turn — refresh it through the bus, and
 * the trailing auto-title refetch is handled once in `QueryBusBridge`.
 */
export function SidebarContent({ onNavigate }: { onNavigate?: () => void }) {
  const router = useRouter();
  const pathname = usePathname();
  const { data: conversations = [], isError } = useQuery(conversationsQuery());
  // "Is the stack up?" is one question, so a failure from either the list or
  // a new-chat POST raises the same banner. Each new-chat attempt starts
  // from a clean slate; the list clears itself on any successful refetch.
  const [newChatFailed, setNewChatFailed] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<ConversationSummary | null>(null);
  const [deleteBusy, setDeleteBusy] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const unreachable = isError || newChatFailed;

  /** Close the hosting drawer (if any) before pushing the route. */
  function go(href: string) {
    onNavigate?.();
    router.push(href);
  }

  async function handleNewChat() {
    setNewChatFailed(false);
    try {
      const created = await createConversation();
      notifyConversationsChanged();
      go(`/c/${created.conversation_id}`);
    } catch {
      setNewChatFailed(true);
    }
  }

  async function confirmDelete() {
    if (deleteTarget === null || deleteBusy) return;
    setDeleteBusy(true);
    setDeleteError(null);
    try {
      await deleteConversation(deleteTarget.conversation_id);
      const wasActive = pathname === `/c/${deleteTarget.conversation_id}`;
      setDeleteTarget(null);
      notifyConversationsChanged(); // the bridge refetches the list for us
      if (wasActive) go("/");
    } catch (failure) {
      setDeleteError(failure instanceof ApiError ? failure.message : String(failure));
    } finally {
      setDeleteBusy(false);
    }
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex flex-col gap-3 px-3 pt-4 pb-3">
        <button
          type="button"
          onClick={() => go("/")}
          className="w-fit rounded-sm px-1 font-heading text-xl leading-none font-normal italic"
        >
          Varagity
        </button>
        <Button size="sm" className="w-full" onClick={() => void handleNewChat()}>
          <PlusIcon aria-hidden />
          New chat
        </Button>
      </div>

      <nav
        aria-label="Conversations"
        className="min-h-0 flex-1 overflow-y-auto px-2 pb-2 scroll-fade-y"
      >
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
                    onClick={() => go(href)}
                    title={conversation.title}
                    aria-current={active ? "page" : undefined}
                    className={cn(
                      "relative w-full truncate rounded-md py-1.5 pr-8 pl-3 text-left text-sm transition-colors",
                      active
                        ? "bg-sidebar-accent text-sidebar-accent-foreground before:absolute before:top-1/2 before:left-1 before:h-3.5 before:w-0.5 before:-translate-y-1/2 before:rounded-full before:bg-primary"
                        : "text-sidebar-foreground/80 hover:bg-sidebar-accent/60 hover:text-sidebar-foreground",
                    )}
                  >
                    {conversation.title}
                  </button>
                  <button
                    type="button"
                    aria-label={`Delete ${conversation.title}`}
                    onClick={() => {
                      setDeleteTarget(conversation);
                      setDeleteError(null);
                    }}
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

      <div className="flex flex-col gap-1 border-t border-border p-2">
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="sm"
            className={cn(
              "flex-1 justify-start",
              pathname === "/corpus" &&
                "bg-sidebar-accent text-sidebar-accent-foreground",
            )}
            onClick={() => go("/corpus")}
          >
            <DatabaseIcon aria-hidden />
            Corpus
          </Button>
          {/* Only the always-mounted desktop instance answers the ⌘K palette,
              so a palette open never raises two dialogs. */}
          <SettingsDrawer openOnBusEvent={onNavigate === undefined} />
        </div>
        <div className="flex items-center justify-between pr-1 pl-0.5">
          <ThemeToggle />
          <span className="text-[10px] text-muted-foreground">
            local · single-user
          </span>
        </div>
      </div>

      <AlertDialog
        open={deleteTarget !== null}
        onOpenChange={(open) => {
          if (!open) setDeleteTarget(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete “{deleteTarget?.title}”?</AlertDialogTitle>
            <AlertDialogDescription>
              The conversation and its messages are removed permanently,
              including their evidence snapshots.
            </AlertDialogDescription>
          </AlertDialogHeader>
          {deleteError && (
            <p role="alert" className="text-sm text-destructive">
              {deleteError}
            </p>
          )}
          <AlertDialogFooter>
            <AlertDialogClose render={<Button variant="outline" />}>
              Cancel
            </AlertDialogClose>
            <Button
              variant="destructive"
              disabled={deleteBusy}
              onClick={() => void confirmDelete()}
            >
              {deleteBusy ? "Deleting…" : "Delete"}
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
