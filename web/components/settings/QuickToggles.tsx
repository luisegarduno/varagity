"use client";

import { useState } from "react";

import {
  settingChoices,
  settingValue,
  useSettingsCatalog,
} from "@/components/settings/use-settings";
import { ApiError } from "@/lib/api";

/**
 * The composer's quick-toggles (spec_v2 §4.7): retrieval method, rerank
 * on/off, and chat model type. They read/write the same persisted override
 * layer as the drawer — a change here is a real `PATCH /api/settings`, so
 * it applies to the next question and shows up everywhere.
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
    <div className="mx-auto flex w-full max-w-3xl flex-wrap items-center gap-x-4 gap-y-1 px-1 pt-1.5 text-xs text-muted-foreground">
      <label className="flex items-center gap-1.5">
        retrieval
        <select
          aria-label="Retrieval method"
          value={String(method ?? "")}
          onChange={(event) => void set("RETRIEVAL_METHOD", event.target.value)}
          className="h-6 rounded-md border border-border bg-background px-1 text-xs"
        >
          {settingChoices(catalog, "RETRIEVAL_METHOD").map((choice) => (
            <option key={choice} value={choice}>
              {choice}
            </option>
          ))}
        </select>
      </label>

      <label className="flex items-center gap-1.5" title="Kill switch: with method=reranked off, it degrades to the base method">
        <input
          type="checkbox"
          aria-label="Rerank enabled"
          checked={rerankEnabled}
          onChange={(event) => void set("RERANK_ENABLED", event.target.checked)}
          className="size-3.5"
        />
        rerank
      </label>

      <label className="flex items-center gap-1.5">
        model
        <select
          aria-label="Chat model type"
          value={String(modelType ?? "")}
          onChange={(event) => void set("CHAT_MODEL_TYPE", event.target.value)}
          className="h-6 rounded-md border border-border bg-background px-1 text-xs"
        >
          {settingChoices(catalog, "CHAT_MODEL_TYPE").map((choice) => (
            <option key={choice} value={choice}>
              {choice}
            </option>
          ))}
        </select>
      </label>

      {error && (
        <span role="alert" className="text-destructive">
          {error}
        </span>
      )}
    </div>
  );
}
