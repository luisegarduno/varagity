"use client";

import { SendIcon, SquareIcon } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { QuickToggles } from "@/components/settings/QuickToggles";
import { Button } from "@/components/ui/button";
import { Kbd } from "@/components/ui/kbd";
import { Textarea } from "@/components/ui/textarea";
import { onFocusComposer } from "@/lib/ui-bus";

/**
 * The question input: Enter sends (Shift+Enter for a newline), the send
 * button flips to Stop while a stream is open (aborting the fetch, which
 * cancels generation server-side between tokens). ArrowUp in the empty
 * field recalls the last question for editing; the ⌘K palette can focus
 * the field via the ui-bus. The quick-toggles underneath write the
 * persisted override layer (spec_v2 §4.7), so they apply to the next
 * question.
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

  // Take focus on mount (a fresh conversation is for typing into) unless
  // something else already holds it, and whenever the palette asks.
  useEffect(() => {
    if (document.activeElement === document.body) {
      textareaRef.current?.focus();
    }
    return onFocusComposer(() => textareaRef.current?.focus());
  }, []);

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

  return (
    <div className="border-t border-border bg-background p-4 pt-2">
      <QuickToggles />
      <form
        className="mx-auto mt-1.5 flex w-full max-w-3xl items-end gap-2"
        onSubmit={(event) => {
          event.preventDefault();
          submit();
        }}
      >
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
