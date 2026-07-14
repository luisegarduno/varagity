"use client";

import { BrainIcon, ChevronDownIcon, ChevronRightIcon } from "lucide-react";
import { useState, useSyncExternalStore } from "react";

import { Button } from "@/components/ui/button";
import {
  reasoningDefaultOpen,
  reasoningDefaultOpenServer,
  subscribeDisplayPrefs,
} from "@/lib/display-prefs";

/**
 * The model's `<think>` stream, collapsible (spec_v2 §4.6): auto-open
 * while the reasoning is streaming in, collapsed once the turn finishes
 * — unless the user toggled it (which wins) or the Phase 8 display
 * setting keeps finished traces expanded by default.
 */
export function ReasoningTrace({
  reasoning,
  streaming,
}: {
  reasoning: string;
  streaming: boolean;
}) {
  const [userChoice, setUserChoice] = useState<boolean | null>(null);
  const defaultOpen = useSyncExternalStore(
    subscribeDisplayPrefs,
    reasoningDefaultOpen,
    reasoningDefaultOpenServer,
  );
  if (!reasoning) return null;
  const open = userChoice ?? (streaming || defaultOpen);

  return (
    <div className="mb-2">
      <Button
        variant="ghost"
        size="xs"
        className="-ml-1 text-muted-foreground"
        aria-expanded={open}
        onClick={() => setUserChoice(!open)}
      >
        {open ? <ChevronDownIcon aria-hidden /> : <ChevronRightIcon aria-hidden />}
        <BrainIcon aria-hidden />
        Reasoning
        {streaming && <span className="animate-pulse">…</span>}
      </Button>
      {open && (
        <div className="mt-1 max-h-60 overflow-y-auto rounded-md border border-border bg-muted/40 p-2 font-mono text-xs leading-relaxed whitespace-pre-wrap text-muted-foreground">
          {reasoning}
        </div>
      )}
    </div>
  );
}
