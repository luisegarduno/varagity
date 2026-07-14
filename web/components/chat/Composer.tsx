"use client";

import { SendIcon, SquareIcon } from "lucide-react";
import { useState } from "react";

import { QuickToggles } from "@/components/settings/QuickToggles";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

/**
 * The question input: Enter sends (Shift+Enter for a newline), the send
 * button flips to Stop while a stream is open (aborting the fetch, which
 * cancels generation server-side between tokens). The quick-toggles
 * underneath write the persisted override layer (spec_v2 §4.7), so they
 * apply to the next question.
 */
export function Composer({
  onSend,
  onStop,
  isStreaming,
}: {
  onSend: (query: string) => void;
  onStop: () => void;
  isStreaming: boolean;
}) {
  const [draft, setDraft] = useState("");

  function submit() {
    const query = draft.trim();
    if (!query || isStreaming) return;
    setDraft("");
    onSend(query);
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
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              submit();
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
    </div>
  );
}
