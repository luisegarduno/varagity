"use client";

import { useQuery } from "@tanstack/react-query";
import {
  ChevronRightIcon,
  DatabaseIcon,
  EllipsisVerticalIcon,
  FolderIcon,
  FolderOpenIcon,
  FolderPlusIcon,
  NetworkIcon,
  PlusIcon,
  Trash2Icon,
} from "lucide-react";
import { usePathname, useRouter } from "next/navigation";
import { useState, useSyncExternalStore, type DragEvent } from "react";

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
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import {
  ApiError,
  createConversation,
  createGroup,
  deleteConversation,
  deleteGroup,
  setConversationGroup,
  type ConversationSummary,
  type GroupOut,
} from "@/lib/api";
import {
  activeGroupId,
  CONVERSATION_DRAG_TYPE,
  conversationIdFromPathname,
  groupConversations,
  isGroupExpanded,
} from "@/lib/conversation-groups";
import { notifyConversationsChanged } from "@/lib/conversations-bus";
import {
  developerMode,
  developerModeServer,
  subscribeDisplayPrefs,
} from "@/lib/display-prefs";
import { notifyGroupsChanged } from "@/lib/groups-bus";
import { conversationsQuery, groupsQuery } from "@/lib/queries";
import { cn } from "@/lib/utils";

/** `dropTarget` sentinel for the ungrouped area (group ids are uuid hex). */
const UNGROUPED_DROP = "__ungrouped__";

/**
 * The navigation rail's inner content — wordmark, "new chat" / "new group",
 * the grouped conversation list, and the Corpus/Settings/theme footer.
 * Rendered by the persistent desktop `Sidebar` and by the mobile navigation
 * drawer (`MobileTopBar`), which passes `onNavigate` so the drawer can close
 * before a route change.
 *
 * Groups render folder-style, collapsed by default; the group holding the
 * active conversation opens itself, and an explicit toggle always wins
 * (`lib/conversation-groups.ts`). Conversations move via drag-and-drop onto
 * a group (or the ungrouped area), with the per-row ⋯ menu as the
 * keyboard/touch path for the same move.
 *
 * The list is the shared `conversations` query, so mutations anywhere —
 * here, the ⌘K palette, a persisted turn — refresh it through the bus, and
 * the trailing auto-title refetch is handled once in `QueryBusBridge`.
 */
export function SidebarContent({ onNavigate }: { onNavigate?: () => void }) {
  const router = useRouter();
  const pathname = usePathname();
  const { data: conversations = [], isError } = useQuery(conversationsQuery());
  const { data: groups = [] } = useQuery(groupsQuery());
  // Developer mode (default on) cosmetically gates the Map entry point; the
  // drawer's switch and this button share the store, so they toggle in sync.
  const devMode = useSyncExternalStore(
    subscribeDisplayPrefs,
    developerMode,
    developerModeServer,
  );
  // "Is the stack up?" is one question, so a failure from either the list or
  // a new-chat POST raises the same banner. Each new-chat attempt starts
  // from a clean slate; the list clears itself on any successful refetch.
  const [newChatFailed, setNewChatFailed] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<ConversationSummary | null>(null);
  const [deleteBusy, setDeleteBusy] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  // Explicit expand/collapse toggles, by group id — the absence of an entry
  // means "derive it" (collapsed unless the group holds the active chat).
  const [expandedOverrides, setExpandedOverrides] = useState<
    Readonly<Record<string, boolean>>
  >({});
  const [newGroupOpen, setNewGroupOpen] = useState(false);
  const [newGroupName, setNewGroupName] = useState("");
  const [newGroupBusy, setNewGroupBusy] = useState(false);
  const [newGroupError, setNewGroupError] = useState<string | null>(null);
  const [deleteGroupTarget, setDeleteGroupTarget] = useState<GroupOut | null>(null);
  const [deleteGroupBusy, setDeleteGroupBusy] = useState(false);
  const [deleteGroupError, setDeleteGroupError] = useState<string | null>(null);
  const [dragConversationId, setDragConversationId] = useState<string | null>(null);
  const [dropTarget, setDropTarget] = useState<string | null>(null);
  const [moveError, setMoveError] = useState<string | null>(null);

  const unreachable = isError || newChatFailed;

  const { sections, ungrouped } = groupConversations(groups, conversations);
  const activeId = conversationIdFromPathname(pathname);
  const activeGroup = activeGroupId(sections, activeId);
  const dragging = dragConversationId !== null;

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

  async function confirmCreateGroup() {
    const name = newGroupName.trim();
    if (name === "" || newGroupBusy) return;
    setNewGroupBusy(true);
    setNewGroupError(null);
    try {
      const created = await createGroup(name);
      setNewGroupOpen(false);
      setNewGroupName("");
      // Open the new (empty) folder so its drop target is visible.
      setExpandedOverrides((current) => ({ ...current, [created.group_id]: true }));
      notifyGroupsChanged();
    } catch (failure) {
      setNewGroupError(failure instanceof ApiError ? failure.message : String(failure));
    } finally {
      setNewGroupBusy(false);
    }
  }

  async function confirmDeleteGroup() {
    if (deleteGroupTarget === null || deleteGroupBusy) return;
    setDeleteGroupBusy(true);
    setDeleteGroupError(null);
    try {
      await deleteGroup(deleteGroupTarget.group_id);
      setDeleteGroupTarget(null);
      notifyGroupsChanged();
      // Members were detached server-side — refresh their group_id too.
      notifyConversationsChanged();
    } catch (failure) {
      setDeleteGroupError(
        failure instanceof ApiError ? failure.message : String(failure),
      );
    } finally {
      setDeleteGroupBusy(false);
    }
  }

  async function moveConversation(conversationId: string, groupId: string | null) {
    const current = conversations.find(
      (conversation) => conversation.conversation_id === conversationId,
    );
    if (!current || (current.group_id ?? null) === groupId) return;
    try {
      await setConversationGroup(conversationId, groupId);
      notifyConversationsChanged();
    } catch (failure) {
      setMoveError(failure instanceof ApiError ? failure.message : String(failure));
    }
  }

  function toggleGroup(groupId: string) {
    setExpandedOverrides((current) => ({
      ...current,
      [groupId]: !isGroupExpanded(groupId, current, activeGroup),
    }));
  }

  function handleRowDragStart(
    event: DragEvent<HTMLLIElement>,
    conversation: ConversationSummary,
  ) {
    event.dataTransfer.setData(CONVERSATION_DRAG_TYPE, conversation.conversation_id);
    event.dataTransfer.effectAllowed = "move";
    setDragConversationId(conversation.conversation_id);
    setMoveError(null);
  }

  function handleRowDragEnd() {
    setDragConversationId(null);
    setDropTarget(null);
  }

  /** Drop-zone handlers for one target (`groupId: null` = ungroup). */
  function dropHandlers(target: string, groupId: string | null) {
    return {
      onDragOver: (event: DragEvent<HTMLElement>) => {
        if (!event.dataTransfer.types.includes(CONVERSATION_DRAG_TYPE)) return;
        event.preventDefault();
        event.dataTransfer.dropEffect = "move";
        setDropTarget(target);
      },
      onDragLeave: (event: DragEvent<HTMLElement>) => {
        if (event.currentTarget.contains(event.relatedTarget as Node | null)) return;
        setDropTarget((current) => (current === target ? null : current));
      },
      onDrop: (event: DragEvent<HTMLElement>) => {
        const conversationId = event.dataTransfer.getData(CONVERSATION_DRAG_TYPE);
        if (!conversationId) return;
        event.preventDefault();
        setDropTarget(null);
        setDragConversationId(null);
        void moveConversation(conversationId, groupId);
      },
    };
  }

  function renderRow(conversation: ConversationSummary) {
    return (
      <ConversationRow
        key={conversation.conversation_id}
        conversation={conversation}
        active={pathname === `/c/${conversation.conversation_id}`}
        dragging={dragConversationId === conversation.conversation_id}
        groups={groups}
        onOpen={() => go(`/c/${conversation.conversation_id}`)}
        onDelete={() => {
          setDeleteTarget(conversation);
          setDeleteError(null);
        }}
        onMove={(groupId) => void moveConversation(conversation.conversation_id, groupId)}
        onDragStart={(event) => handleRowDragStart(event, conversation)}
        onDragEnd={handleRowDragEnd}
      />
    );
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
        <div className="flex gap-2">
          <Button
            size="sm"
            className="min-w-0 flex-1"
            onClick={() => void handleNewChat()}
          >
            <PlusIcon aria-hidden />
            New chat
          </Button>
          <Button
            size="sm"
            variant="outline"
            className="min-w-0 flex-1"
            onClick={() => setNewGroupOpen(true)}
          >
            <FolderPlusIcon aria-hidden />
            New group
          </Button>
        </div>
      </div>

      <nav
        aria-label="Conversations"
        className="min-h-0 flex-1 overflow-y-auto px-2 pb-2 scroll-fade-y"
      >
        {unreachable ? (
          <p className="px-2 py-4 text-xs text-muted-foreground">
            API unreachable — is the stack up? (docker compose up -d)
          </p>
        ) : conversations.length === 0 && groups.length === 0 ? (
          <p className="px-2 py-4 text-xs text-muted-foreground">
            No conversations yet.
          </p>
        ) : (
          <>
            {moveError && (
              <p role="alert" className="px-2 py-1 text-xs text-destructive">
                {moveError}
              </p>
            )}
            {sections.length > 0 && (
              <ul className="flex flex-col gap-0.5 pb-1">
                {sections.map((section) => {
                  const groupId = section.group.group_id;
                  const expanded = isGroupExpanded(
                    groupId,
                    expandedOverrides,
                    activeGroup,
                  );
                  return (
                    <li
                      key={groupId}
                      className={cn(
                        "rounded-md transition-colors",
                        dropTarget === groupId &&
                          "bg-sidebar-accent/50 ring-1 ring-primary/40",
                      )}
                      {...dropHandlers(groupId, groupId)}
                    >
                      <div className="group relative">
                        <button
                          type="button"
                          aria-expanded={expanded}
                          onClick={() => toggleGroup(groupId)}
                          title={section.group.name}
                          className="flex w-full min-w-0 items-center gap-1.5 rounded-md py-1.5 pr-8 pl-2 text-left text-sm text-sidebar-foreground/80 transition-colors hover:bg-sidebar-accent/60 hover:text-sidebar-foreground"
                        >
                          <ChevronRightIcon
                            aria-hidden
                            className={cn(
                              "size-3.5 shrink-0 text-muted-foreground transition-transform",
                              expanded && "rotate-90",
                            )}
                          />
                          {expanded ? (
                            <FolderOpenIcon
                              aria-hidden
                              className="size-3.5 shrink-0 text-muted-foreground"
                            />
                          ) : (
                            <FolderIcon
                              aria-hidden
                              className="size-3.5 shrink-0 text-muted-foreground"
                            />
                          )}
                          <span className="min-w-0 flex-1 truncate">
                            {section.group.name}
                          </span>
                          <span className="shrink-0 text-[10px] text-muted-foreground tabular-nums">
                            {section.conversations.length}
                          </span>
                        </button>
                        <button
                          type="button"
                          aria-label={`Delete group ${section.group.name}`}
                          onClick={() => {
                            setDeleteGroupTarget(section.group);
                            setDeleteGroupError(null);
                          }}
                          className="absolute top-1/2 right-1.5 -translate-y-1/2 rounded p-1 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100 hover:text-destructive focus-visible:opacity-100"
                        >
                          <Trash2Icon className="size-3.5" />
                        </button>
                      </div>
                      {expanded &&
                        (section.conversations.length === 0 ? (
                          <p className="py-1 pr-2 pl-9 text-xs text-muted-foreground">
                            Drag chats here
                          </p>
                        ) : (
                          <ul className="flex flex-col gap-0.5 pl-3">
                            {section.conversations.map(renderRow)}
                          </ul>
                        ))}
                    </li>
                  );
                })}
              </ul>
            )}
            <div
              className={cn(
                "rounded-md transition-colors",
                dragging &&
                  dropTarget === UNGROUPED_DROP &&
                  "bg-sidebar-accent/50 ring-1 ring-primary/40",
              )}
              {...dropHandlers(UNGROUPED_DROP, null)}
            >
              {ungrouped.length > 0 ? (
                <ul className="flex flex-col gap-0.5">{ungrouped.map(renderRow)}</ul>
              ) : dragging ? (
                <p className="px-2 py-2 text-xs text-muted-foreground">
                  Drop here to ungroup
                </p>
              ) : null}
            </div>
          </>
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
          {devMode && (
            <Button
              variant="ghost"
              size="sm"
              className={cn(
                "justify-start",
                pathname === "/map" &&
                  "bg-sidebar-accent text-sidebar-accent-foreground",
              )}
              onClick={() => go("/map")}
            >
              <NetworkIcon aria-hidden />
              Map
            </Button>
          )}
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

      <Dialog
        open={newGroupOpen}
        onOpenChange={(open) => {
          setNewGroupOpen(open);
          if (!open) {
            setNewGroupName("");
            setNewGroupError(null);
          }
        }}
      >
        <DialogContent>
          <form
            className="grid gap-4"
            onSubmit={(event) => {
              event.preventDefault();
              void confirmCreateGroup();
            }}
          >
            <DialogHeader>
              <DialogTitle>New group</DialogTitle>
              <DialogDescription>
                Groups organize the sidebar. Drag a conversation onto a group
                to file it.
              </DialogDescription>
            </DialogHeader>
            <Input
              value={newGroupName}
              onChange={(event) => setNewGroupName(event.target.value)}
              placeholder="Group name"
              aria-label="Group name"
              maxLength={200}
              autoFocus
            />
            {newGroupError && (
              <p role="alert" className="text-sm text-destructive">
                {newGroupError}
              </p>
            )}
            <DialogFooter>
              <DialogClose render={<Button variant="outline" type="button" />}>
                Cancel
              </DialogClose>
              <Button
                type="submit"
                disabled={newGroupName.trim() === "" || newGroupBusy}
              >
                {newGroupBusy ? "Creating…" : "Create"}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      <AlertDialog
        open={deleteGroupTarget !== null}
        onOpenChange={(open) => {
          if (!open) setDeleteGroupTarget(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              Delete group “{deleteGroupTarget?.name}”?
            </AlertDialogTitle>
            <AlertDialogDescription>
              Its conversations are kept — they move back to the ungrouped
              list.
            </AlertDialogDescription>
          </AlertDialogHeader>
          {deleteGroupError && (
            <p role="alert" className="text-sm text-destructive">
              {deleteGroupError}
            </p>
          )}
          <AlertDialogFooter>
            <AlertDialogClose render={<Button variant="outline" />}>
              Cancel
            </AlertDialogClose>
            <Button
              variant="destructive"
              disabled={deleteGroupBusy}
              onClick={() => void confirmDeleteGroup()}
            >
              {deleteGroupBusy ? "Deleting…" : "Delete"}
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

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

/**
 * One conversation row: navigates on click, drags as a whole (the HTML5
 * payload is the conversation id), and carries two hover affordances — the
 * ⋯ move menu (the keyboard/touch path to grouping) and delete.
 */
function ConversationRow({
  conversation,
  active,
  dragging,
  groups,
  onOpen,
  onDelete,
  onMove,
  onDragStart,
  onDragEnd,
}: {
  conversation: ConversationSummary;
  active: boolean;
  dragging: boolean;
  groups: GroupOut[];
  onOpen: () => void;
  onDelete: () => void;
  onMove: (groupId: string | null) => void;
  onDragStart: (event: DragEvent<HTMLLIElement>) => void;
  onDragEnd: () => void;
}) {
  return (
    <li
      draggable
      onDragStart={onDragStart}
      onDragEnd={onDragEnd}
      className={cn("group relative", dragging && "opacity-50")}
    >
      <button
        type="button"
        onClick={onOpen}
        title={conversation.title}
        aria-current={active ? "page" : undefined}
        className={cn(
          "relative w-full truncate rounded-md py-1.5 pr-13 pl-3 text-left text-sm transition-colors",
          active
            ? "bg-sidebar-accent text-sidebar-accent-foreground before:absolute before:top-1/2 before:left-1 before:h-3.5 before:w-0.5 before:-translate-y-1/2 before:rounded-full before:bg-primary"
            : "text-sidebar-foreground/80 hover:bg-sidebar-accent/60 hover:text-sidebar-foreground",
        )}
      >
        {conversation.title}
      </button>
      <span className="absolute top-1/2 right-1.5 flex -translate-y-1/2 items-center">
        <DropdownMenu>
          <DropdownMenuTrigger
            render={
              <button
                type="button"
                aria-label={`Move ${conversation.title} to a group`}
                className="rounded p-1 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100 hover:text-foreground focus-visible:opacity-100 data-popup-open:opacity-100"
              />
            }
          >
            <EllipsisVerticalIcon className="size-3.5" />
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-44">
            {/* GroupLabel throws outside a Menu.Group — the wrapper is load-bearing. */}
            <DropdownMenuGroup>
              <DropdownMenuLabel>Move to group</DropdownMenuLabel>
              {groups.length === 0 && (
                <DropdownMenuItem disabled>No groups yet</DropdownMenuItem>
              )}
              {groups.map((group) => (
                <DropdownMenuItem
                  key={group.group_id}
                  disabled={group.group_id === conversation.group_id}
                  onClick={() => onMove(group.group_id)}
                >
                  <FolderIcon aria-hidden />
                  <span className="truncate">{group.name}</span>
                </DropdownMenuItem>
              ))}
            </DropdownMenuGroup>
            {conversation.group_id != null && (
              <>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={() => onMove(null)}>
                  Remove from group
                </DropdownMenuItem>
              </>
            )}
          </DropdownMenuContent>
        </DropdownMenu>
        <button
          type="button"
          aria-label={`Delete ${conversation.title}`}
          onClick={onDelete}
          className="rounded p-1 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100 hover:text-destructive focus-visible:opacity-100"
        >
          <Trash2Icon className="size-3.5" />
        </button>
      </span>
    </li>
  );
}
