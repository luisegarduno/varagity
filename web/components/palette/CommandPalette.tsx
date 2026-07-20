"use client";

import { Autocomplete } from "@base-ui/react/autocomplete";
import { Dialog as DialogPrimitive } from "@base-ui/react/dialog";
import { useQuery } from "@tanstack/react-query";
import { SearchIcon } from "lucide-react";
import { useRouter } from "next/navigation";
import { useTheme } from "next-themes";
import * as React from "react";

import { Kbd } from "@/components/ui/kbd";
import { useMountEffect } from "@/hooks/use-mount-effect";
import { createConversation, type ConversationSummary } from "@/lib/api";
import { notifyConversationsChanged } from "@/lib/conversations-bus";
import {
  developerMode,
  developerModeServer,
  subscribeDisplayPrefs,
} from "@/lib/display-prefs";
import {
  filterCommands,
  groupCommands,
  type PaletteCommand,
  type PaletteGroup,
} from "@/lib/palette";
import { conversationsQuery } from "@/lib/queries";
import {
  notifyFocusComposer,
  notifyOpenSettings,
  notifyToggleEvidence,
} from "@/lib/ui-bus";

/** How many conversations the un-queried list shows (filtering sees all). */
const RECENT_CONVERSATIONS_LIMIT = 8;

/**
 * The static command inventory. Serializable rows only — `executeCommand`
 * maps each id onto its action, keyed by the `kind:argument` id shape.
 */
const STATIC_COMMANDS: PaletteCommand[] = [
  {
    id: "action:new-chat",
    label: "New chat",
    group: "Actions",
    keywords: ["create", "start", "conversation"],
  },
  {
    id: "action:focus-composer",
    label: "Focus composer",
    group: "Actions",
    keywords: ["input", "message", "type", "ask"],
  },
  {
    id: "action:toggle-evidence",
    label: "Toggle evidence panel",
    group: "Actions",
    keywords: ["sources", "citations", "provenance", "rail", "trace"],
  },
  {
    id: "navigate:corpus",
    label: "Corpus",
    group: "Navigate",
    keywords: ["documents", "files", "upload", "ingest"],
  },
  {
    id: "navigate:map",
    label: "Codebase Map",
    group: "Navigate",
    keywords: ["architecture", "graph", "diagram", "developer", "system"],
  },
  {
    id: "navigate:settings",
    label: "Settings",
    group: "Navigate",
    keywords: ["preferences", "configuration", "options", "drawer"],
  },
  {
    id: "theme:light",
    label: "Theme: Light",
    group: "Appearance",
    keywords: ["theme", "appearance", "mode", "day"],
  },
  {
    id: "theme:dark",
    label: "Theme: Dark",
    group: "Appearance",
    keywords: ["theme", "appearance", "mode", "night"],
  },
  {
    id: "theme:system",
    label: "Theme: System",
    group: "Appearance",
    keywords: ["theme", "appearance", "mode", "auto"],
  },
];

/** Compact "how long ago" label for a conversation's last activity. */
function relativeTimeLabel(iso: string): string | undefined {
  const time = Date.parse(iso);
  if (Number.isNaN(time)) return undefined;
  const minutes = Math.floor((Date.now() - time) / 60_000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days}d ago`;
  return new Date(time).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

/** Map one conversation summary onto a palette command row. */
function toConversationCommand(
  conversation: ConversationSummary,
): PaletteCommand {
  const title = conversation.title.trim() || "Untitled";
  return {
    id: `conversation:${conversation.conversation_id}`,
    label: title,
    group: "Conversations",
    keywords: title.toLowerCase().split(/\s+/).filter(Boolean),
    hint: relativeTimeLabel(conversation.updated_at),
  };
}

/**
 * The ⌘K command palette (Phase 9): a top-aligned Dialog hosting an inline
 * Autocomplete over actions, navigation, appearance, and recent
 * conversations — the Base UI "command palette" recipe in house styling.
 *
 * Fully self-contained: it registers its own global ⌘K / Ctrl+K toggle and
 * needs only the root providers (ThemeProvider, the app router), so the
 * layout renders `<CommandPalette />` once and is done. Filtering is the
 * pure `lib/palette.ts` logic, wired in via `filteredItems` + `filter={null}`
 * (Base UI's internal filter stays out of the way).
 */
export function CommandPalette() {
  const router = useRouter();
  const { setTheme } = useTheme();
  const hintsId = React.useId();
  const devMode = React.useSyncExternalStore(
    subscribeDisplayPrefs,
    developerMode,
    developerModeServer,
  );

  const [open, setOpen] = React.useState(false);
  const [query, setQuery] = React.useState("");
  // The selected command's action, parked until the dialog finishes closing.
  const pendingActionRef = React.useRef<(() => void) | null>(null);

  // Global shortcut: ⌘K on mac, Ctrl+K elsewhere — also while inputs are
  // focused (preventDefault stops the browser default). Registered once:
  // the updater reads `open`, so the listener never re-binds, and state
  // changes stay inside the event callback.
  useMountEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.defaultPrevented) return;
      if (!(event.metaKey || event.ctrlKey)) return;
      if (event.key.toLowerCase() !== "k") return;
      event.preventDefault();
      // Unconditional: a close is about to discard the query anyway, and
      // every open path resets it (see `handleOpenChange`).
      setQuery("");
      setOpen((current) => !current);
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  });

  // The Conversations group. Only fetched once the palette opens, and
  // shared with the sidebar's list — which usually has it warm already, so
  // an open paints instantly. On failure the group is silently omitted.
  const { data: conversations } = useQuery({
    ...conversationsQuery(),
    enabled: open,
  });

  const executeCommand = React.useCallback(
    (command: PaletteCommand) => {
      const separator = command.id.indexOf(":");
      const kind = separator === -1 ? command.id : command.id.slice(0, separator);
      const argument = separator === -1 ? "" : command.id.slice(separator + 1);
      switch (kind) {
        case "action":
          if (argument === "new-chat") {
            createConversation().then(
              (created) => {
                notifyConversationsChanged();
                router.push(`/c/${created.conversation_id}`);
              },
              () => {
                // API unreachable — the sidebar already surfaces that state.
              },
            );
          } else if (argument === "focus-composer") {
            notifyFocusComposer();
          } else if (argument === "toggle-evidence") {
            notifyToggleEvidence();
          }
          break;
        case "navigate":
          if (argument === "corpus") {
            router.push("/corpus");
          } else if (argument === "map") {
            router.push("/map");
          } else if (argument === "settings") {
            notifyOpenSettings();
          }
          break;
        case "theme":
          if (
            argument === "light" ||
            argument === "dark" ||
            argument === "system"
          ) {
            setTheme(argument);
          }
          break;
        case "conversation":
          router.push(`/c/${argument}`);
          break;
      }
    },
    [router, setTheme],
  );

  // Close-then-act: park the action and close. It runs from
  // `onOpenChangeComplete(false)` below, i.e. only after the exit
  // transition settles and Base UI has restored focus — otherwise the
  // focus restore would immediately steal focus back from the composer or
  // the settings drawer. The palette lives in the root layout, so
  // navigation is never swallowed by an unmount.
  function runCommand(command: PaletteCommand) {
    pendingActionRef.current = () => executeCommand(command);
    setOpen(false);
  }

  function handleOpenChange(nextOpen: boolean) {
    if (nextOpen) setQuery("");
    setOpen(nextOpen);
  }

  function handleOpenChangeComplete(nextOpen: boolean) {
    if (nextOpen) return;
    const pending = pendingActionRef.current;
    pendingActionRef.current = null;
    if (pending) window.setTimeout(pending, 0);
  }

  const conversationCommands = React.useMemo(
    () => (conversations ?? []).map(toConversationCommand),
    [conversations],
  );

  // Developer mode (default on) hides the Codebase Map command. It must drop
  // from *both* memos below — Base UI keeps its own item registry from
  // `items`, so filtering only `filteredItems` would leave the hidden command
  // matchable via the autocomplete.
  const staticCommands = React.useMemo(
    () =>
      devMode
        ? STATIC_COMMANDS
        : STATIC_COMMANDS.filter((command) => command.id !== "navigate:map"),
    [devMode],
  );

  // The full universe (for Base UI's `items`) and the query-filtered view
  // (`filteredItems`). Un-queried, the Conversations group is capped to the
  // most recent few; a query searches everything fetched.
  const allGroups = React.useMemo(
    () => groupCommands([...staticCommands, ...conversationCommands]),
    [staticCommands, conversationCommands],
  );
  const visibleGroups = React.useMemo(() => {
    const visible =
      query.trim() === ""
        ? [
            ...staticCommands,
            ...conversationCommands.slice(0, RECENT_CONVERSATIONS_LIMIT),
          ]
        : filterCommands(
            [...staticCommands, ...conversationCommands],
            query,
          );
    return groupCommands(visible);
  }, [staticCommands, conversationCommands, query]);

  return (
    <DialogPrimitive.Root
      open={open}
      onOpenChange={handleOpenChange}
      onOpenChangeComplete={handleOpenChangeComplete}
    >
      <DialogPrimitive.Portal>
        <DialogPrimitive.Backdrop
          data-slot="command-palette-overlay"
          className="fixed inset-0 isolate z-50 bg-black/20 transition-opacity duration-200 supports-backdrop-filter:backdrop-blur-xs data-starting-style:opacity-0 data-ending-style:opacity-0 motion-reduce:transition-none dark:bg-black/40"
        />
        <DialogPrimitive.Popup
          data-slot="command-palette"
          aria-label="Command palette"
          className="fixed top-[18%] left-1/2 z-50 w-full max-w-[calc(100%-2rem)] -translate-x-1/2 overflow-hidden rounded-xl bg-popover text-popover-foreground shadow-lg ring-1 ring-foreground/10 transition-[opacity,scale] duration-200 ease-[cubic-bezier(0.16,1,0.3,1)] outline-none sm:max-w-lg data-starting-style:scale-[0.98] data-starting-style:opacity-0 data-ending-style:scale-[0.98] data-ending-style:opacity-0 motion-reduce:transition-none"
        >
          <Autocomplete.Root
            open
            inline
            items={allGroups}
            filteredItems={visibleGroups}
            filter={null}
            autoHighlight="always"
            keepHighlight
            value={query}
            onValueChange={(value: string) => setQuery(value)}
            itemToStringValue={(command: PaletteCommand) => command.label}
          >
            <div
              data-slot="command-palette-input-row"
              className="flex items-center gap-2.5 px-3.5"
            >
              <SearchIcon
                aria-hidden
                className="size-4 shrink-0 text-muted-foreground"
              />
              <Autocomplete.Input
                aria-label="Type a command or search"
                aria-describedby={hintsId}
                // Base UI computes aria-expanded only in popup (non-inline)
                // mode, but stamps role="combobox" regardless — without this
                // the inline recipe trips axe's critical aria-required-attr.
                // The inline list is always rendered while the palette is
                // open, so "true" is accurate (extra props merge rightmost-
                // wins onto the DOM input).
                aria-expanded="true"
                placeholder="Type a command or search…"
                className="h-12 w-full min-w-0 bg-transparent text-base outline-none placeholder:text-muted-foreground"
              />
            </div>
            <DialogPrimitive.Close className="sr-only">
              Close command palette
            </DialogPrimitive.Close>

            <div className="border-t border-border">
              <Autocomplete.Empty>
                <div className="flex min-h-24 items-center justify-center px-4 py-6 text-center font-heading text-base font-normal text-muted-foreground">
                  No matching commands
                </div>
              </Autocomplete.Empty>
              <Autocomplete.List className="max-h-80 scroll-py-1 overflow-y-auto overscroll-contain p-1 outline-none data-empty:p-0">
                {(group: PaletteGroup) => (
                  <Autocomplete.Group
                    key={group.group}
                    items={group.items}
                    className="not-last:mb-1"
                  >
                    <Autocomplete.GroupLabel className="px-2 pt-2 pb-1 text-[0.6875rem] font-medium tracking-[0.08em] text-muted-foreground uppercase select-none">
                      {group.group}
                    </Autocomplete.GroupLabel>
                    <Autocomplete.Collection>
                      {(command: PaletteCommand) => (
                        <Autocomplete.Item
                          key={command.id}
                          value={command}
                          onClick={() => runCommand(command)}
                          className="group/palette-item flex scroll-my-1 cursor-default items-center gap-2 rounded-md px-2 py-1.5 text-sm outline-none select-none data-highlighted:bg-accent data-highlighted:text-accent-foreground"
                        >
                          <span className="min-w-0 flex-1 truncate">
                            {command.label}
                          </span>
                          {command.hint ? (
                            <span className="shrink-0 text-xs text-muted-foreground group-data-highlighted/palette-item:text-accent-foreground">
                              {command.hint}
                            </span>
                          ) : null}
                        </Autocomplete.Item>
                      )}
                    </Autocomplete.Collection>
                  </Autocomplete.Group>
                )}
              </Autocomplete.List>
            </div>

            <div
              data-slot="command-palette-footer"
              className="flex items-center gap-3 border-t border-border px-3.5 py-2 text-[0.6875rem] text-muted-foreground select-none"
            >
              <span id={hintsId} className="sr-only">
                Use the arrow keys to navigate and Enter to run the
                highlighted command.
              </span>
              <span aria-hidden className="flex items-center gap-1">
                <Kbd>↑</Kbd>
                <Kbd>↓</Kbd> navigate
              </span>
              <span aria-hidden className="flex items-center gap-1">
                <Kbd>↵</Kbd> run
              </span>
              <span aria-hidden className="flex items-center gap-1">
                <Kbd>esc</Kbd> close
              </span>
            </div>
          </Autocomplete.Root>
        </DialogPrimitive.Popup>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}
