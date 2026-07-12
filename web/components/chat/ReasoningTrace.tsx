"use client";

import { BrainIcon, ChevronDownIcon, ChevronRightIcon } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";

/**
 * The model's `<think>` stream, collapsible (spec_v2 §4.6): auto-open
 * while the reasoning is streaming in, collapsed once the turn finishes
 * — unless the user toggled it, which then wins. (A default-open setting
 * arrives with the Phase 8 settings drawer.)
 */
export function ReasoningTrace({
  reasoning,
  streaming,
}: {
  reasoning: string;
  streaming: boolean;
}) {
  const [userChoice, setUserChoice] = useState<boolean | null>(null);
  if (!reasoning) return null;
  const open = userChoice ?? streaming;

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
