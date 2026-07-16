"use client";

import { useQuery } from "@tanstack/react-query";
import {
  FileUpIcon,
  FolderUpIcon,
  PaperclipIcon,
  SendIcon,
  SquareIcon,
  XIcon,
} from "lucide-react";
import { useRef, useState } from "react";

import { QuickToggles } from "@/components/settings/QuickToggles";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Kbd } from "@/components/ui/kbd";
import { Textarea } from "@/components/ui/textarea";
import { useMountEffect } from "@/hooks/use-mount-effect";
import { configQuery } from "@/lib/queries";
import { onFocusComposer } from "@/lib/ui-bus";
import { attachChipLabel, useComposerAttach } from "@/lib/use-upload";
import { cn } from "@/lib/utils";

/**
 * The question input: Enter sends (Shift+Enter for a newline), the send
 * button flips to Stop while a stream is open (aborting the fetch, which
 * cancels generation server-side between tokens). ArrowUp in the empty
 * field recalls the last question for editing; the ⌘K palette can focus
 * the field via the ui-bus. The quick-toggles underneath write the
 * persisted override layer (spec_v2 §4.7), so they apply to the next
 * question.
 *
 * The 📎 menu (spec_v3 §5.3) attaches files *or a folder* straight from
 * here: upload → auto-ingest (`reingest: false`), with progress as a
 * compact chip in this area — not a modal, so typing and sending continue
 * while it runs. `/corpus` remains the full management surface.
 */
export function Composer({
  onSend,
  onStop,
  isStreaming,
  lastUserQuery,
}: {
  onSend: (query: string) => void;
  onStop: () => void;
  isStreaming: boolean;
  /** The most recent user question (`null` when none) — ↑ recalls it. */
  lastUserQuery: string | null;
}) {
  const [draft, setDraft] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const filesInputRef = useRef<HTMLInputElement | null>(null);
  const folderInputRef = useRef<HTMLInputElement | null>(null);

  const { data: config = null } = useQuery(configQuery());
  const { state: attachState, attach, dismiss } = useComposerAttach();
  const chipLabel = attachChipLabel(attachState);
  const attachActive =
    attachState.phase === "uploading" ||
    attachState.phase === "queued" ||
    attachState.phase === "ingesting";
  const attachTerminal =
    attachState.phase === "done" || attachState.phase === "error";

  // Take focus on mount (a fresh conversation is for typing into) unless
  // something else already holds it, and whenever the palette asks.
  useMountEffect(() => {
    if (document.activeElement === document.body) {
      textareaRef.current?.focus();
    }
    return onFocusComposer(() => textareaRef.current?.focus());
  });

  function submit() {
    const query = draft.trim();
    if (!query || isStreaming) return;
    setDraft("");
    onSend(query);
  }

  function recallLastQuery() {
    if (!lastUserQuery) return;
    setDraft(lastUserQuery);
    // The controlled update keeps the caret at 0 — move it to the end
    // once the new value is in the DOM.
    requestAnimationFrame(() => {
      const element = textareaRef.current;
      element?.setSelectionRange(element.value.length, element.value.length);
    });
  }

  function handlePicked(
    input: HTMLInputElement | null,
    options: { folder: boolean },
  ) {
    if (!input) return;
    attach(Array.from(input.files ?? []), { ...options, config });
    input.value = ""; // allow re-picking the same selection
  }

  return (
    <div className="border-t border-border bg-background p-4 pt-2">
      <QuickToggles />
      {chipLabel && (
        <div
          role="status"
          className="mx-auto mt-1.5 flex w-full max-w-3xl flex-wrap items-center gap-2 px-1"
        >
          <span
            className={cn(
              "inline-flex max-w-full items-center gap-1.5 truncate rounded-full border px-2.5 py-1 text-xs",
              attachState.phase === "error"
                ? "border-destructive/30 bg-destructive/5 text-destructive"
                : "border-border bg-muted/40 text-muted-foreground",
            )}
          >
            <PaperclipIcon aria-hidden className="size-3 shrink-0" />
            <span className={cn("truncate", attachActive && "shimmer")}>
              {chipLabel}
            </span>
          </span>
          {attachState.skipped &&
            attachState.skipped !== chipLabel && (
              <span className="text-xs text-muted-foreground">
                {attachState.skipped}
              </span>
            )}
          {attachTerminal && (
            <Button
              type="button"
              variant="ghost"
              size="icon-xs"
              aria-label="Dismiss upload status"
              onClick={dismiss}
            >
              <XIcon />
            </Button>
          )}
        </div>
      )}
      <form
        className="mx-auto mt-1.5 flex w-full max-w-3xl items-end gap-2"
        onSubmit={(event) => {
          event.preventDefault();
          submit();
        }}
      >
        <DropdownMenu>
          <DropdownMenuTrigger
            render={
              <Button
                type="button"
                variant="outline"
                size="icon"
                aria-label="Attach documents to the corpus"
              />
            }
          >
            <PaperclipIcon className="size-4" />
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" side="top">
            <DropdownMenuItem onClick={() => filesInputRef.current?.click()}>
              <FileUpIcon aria-hidden /> Add files…
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => folderInputRef.current?.click()}>
              <FolderUpIcon aria-hidden /> Add folder…
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
        <input
          ref={filesInputRef}
          type="file"
          multiple
          accept={config?.allowed_extensions.join(",")}
          className="hidden"
          onChange={(event) =>
            handlePicked(event.target, { folder: false })
          }
        />
        {/* The directory picker ignores `accept` — the attach flow filters
            client-side and summarizes what it skipped (spec_v3 §5.3). The
            non-standard attributes ride a spread: React's types don't know
            them, browsers do. */}
        <input
          ref={folderInputRef}
          type="file"
          multiple
          className="hidden"
          onChange={(event) => handlePicked(event.target, { folder: true })}
          {...{ webkitdirectory: "", directory: "" }}
        />
        <Textarea
          ref={textareaRef}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              submit();
            } else if (
              event.key === "ArrowUp" &&
              draft === "" &&
              !isStreaming
            ) {
              // Empty field ⇒ the caret is at the start: recall the last
              // question for editing. Inside text, ArrowUp stays native.
              event.preventDefault();
              recallLastQuery();
            }
          }}
          placeholder="Ask about your corpus…"
          aria-label="Question"
          rows={2}
          className="max-h-40 min-h-16 flex-1 resize-none"
        />
        {isStreaming ? (
          <Button
            type="button"
            variant="outline"
            size="icon"
            aria-label="Stop generating"
            onClick={onStop}
          >
            <SquareIcon className="size-4" />
          </Button>
        ) : (
          <Button
            type="submit"
            size="icon"
            aria-label="Send"
            disabled={!draft.trim()}
          >
            <SendIcon className="size-4" />
          </Button>
        )}
      </form>
      <div
        aria-hidden="true"
        className="mx-auto mt-1.5 hidden w-full max-w-3xl flex-wrap items-center gap-x-3 gap-y-1 px-1 text-[11px] text-muted-foreground select-none sm:flex"
      >
        <span className="inline-flex items-center gap-1">
          <Kbd>↵</Kbd> send
        </span>
        <span className="inline-flex items-center gap-1">
          <Kbd>⇧↵</Kbd> newline
        </span>
        <span className="inline-flex items-center gap-1">
          <Kbd>Esc</Kbd> stop
        </span>
        <span className="inline-flex items-center gap-1">
          <Kbd>↑</Kbd> edit last
        </span>
      </div>
    </div>
  );
}
