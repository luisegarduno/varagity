"use client";

import { BrainIcon, ChevronRightIcon } from "lucide-react";
import { useState, useSyncExternalStore } from "react";

import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsiblePanel,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  reasoningDefaultOpen,
  reasoningDefaultOpenServer,
  subscribeDisplayPrefs,
} from "@/lib/display-prefs";
import { cn } from "@/lib/utils";

/**
 * The model's `<think>` stream, collapsible (spec_v2 §4.6): auto-open
 * while the reasoning is streaming in, collapsed once the turn finishes
 * — unless the user toggled it (which wins) or the display
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
    <Collapsible
      open={open}
      onOpenChange={setUserChoice}
      className="mb-2"
    >
      <CollapsibleTrigger
        render={
          <Button
            variant="ghost"
            size="xs"
            className="-ml-1 text-muted-foreground"
          />
        }
      >
        <ChevronRightIcon
          aria-hidden
          className="motion-safe:transition-transform group-aria-expanded/button:rotate-90"
        />
        <BrainIcon aria-hidden />
        <span className={cn(streaming && "shimmer")}>
          {streaming ? "Reasoning…" : "Reasoning trace"}
        </span>
      </CollapsibleTrigger>
      <CollapsiblePanel>
        <div className="mt-1 max-h-60 overflow-y-auto rounded-md border border-border bg-muted/40 p-2 font-mono text-xs leading-relaxed whitespace-pre-wrap text-muted-foreground">
          {reasoning}
        </div>
      </CollapsiblePanel>
    </Collapsible>
  );
}
