"use client";

import { useState } from "react";

import {
  settingChoices,
  settingValue,
  useSettingsCatalog,
} from "@/components/settings/use-settings";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { ApiError } from "@/lib/api";

/**
 * The composer's quick-toggles (spec_v2 §4.7): retrieval method, rerank
 * on/off, and chat model type. They read/write the same persisted override
 * layer as the drawer — a change here is a real `PATCH /api/settings`, so
 * it applies to the next question and shows up everywhere. Styled quiet so
 * the composer stays the hero.
 */
export function QuickToggles() {
  const { catalog, patch } = useSettingsCatalog();
  const [error, setError] = useState<string | null>(null);

  async function set(name: string, value: boolean | string) {
    setError(null);
    try {
      await patch({ [name]: value });
    } catch (failure) {
      setError(failure instanceof ApiError ? failure.message : String(failure));
    }
  }

  if (catalog === null) return null;

  const method = settingValue(catalog, "RETRIEVAL_METHOD");
  const rerankEnabled = settingValue(catalog, "RERANK_ENABLED") === true;
  const modelType = settingValue(catalog, "CHAT_MODEL_TYPE");

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-wrap items-center gap-x-4 gap-y-1.5 px-1 pt-1.5 text-xs text-muted-foreground">
      <QuickSelect
        id="quick-retrieval"
        label="retrieval"
        ariaLabel="Retrieval method"
        value={String(method ?? "")}
        choices={settingChoices(catalog, "RETRIEVAL_METHOD")}
        onChange={(value) => void set("RETRIEVAL_METHOD", value)}
      />

      <label
        className="flex items-center gap-1.5"
        title="Kill switch: with method=reranked off, it degrades to the base method"
      >
        rerank
        <Switch
          aria-label="Rerank enabled"
          checked={rerankEnabled}
          onCheckedChange={(checked) => void set("RERANK_ENABLED", checked)}
        />
      </label>

      <QuickSelect
        id="quick-model"
        label="model"
        ariaLabel="Chat model type"
        value={String(modelType ?? "")}
        choices={settingChoices(catalog, "CHAT_MODEL_TYPE")}
        onChange={(value) => void set("CHAT_MODEL_TYPE", value)}
      />

      {error && (
        <span role="alert" className="text-destructive">
          {error}
        </span>
      )}
    </div>
  );
}

/** One labelled compact select — borderless until interacted with. */
function QuickSelect({
  id,
  label,
  ariaLabel,
  value,
  choices,
  onChange,
}: {
  id: string;
  label: string;
  ariaLabel: string;
  value: string;
  choices: string[];
  onChange: (value: string) => void;
}) {
  return (
    <div className="flex items-center gap-1.5">
      <label htmlFor={id}>{label}</label>
      <Select
        value={value}
        onValueChange={(next) => {
          if (typeof next === "string" && next !== value) onChange(next);
        }}
      >
        <SelectTrigger
          id={id}
          aria-label={ariaLabel}
          className="h-6 w-auto gap-1 border-transparent px-1.5 text-xs text-foreground hover:bg-muted dark:bg-transparent"
        >
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {choices.map((choice) => (
            <SelectItem key={choice} value={choice} className="text-xs">
              {choice}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
