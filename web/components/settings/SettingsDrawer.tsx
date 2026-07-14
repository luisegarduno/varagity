"use client";

import { RotateCcwIcon, SettingsIcon } from "lucide-react";
import { useTheme } from "next-themes";
import { useRouter } from "next/navigation";
import { useEffect, useState, useSyncExternalStore } from "react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { useSettingsCatalog } from "@/components/settings/use-settings";
import { ApiError, getConfig, startIngest, type ConfigResponse } from "@/lib/api";
import {
  reasoningDefaultOpen,
  reasoningDefaultOpenServer,
  setReasoningDefaultOpen,
  subscribeDisplayPrefs,
} from "@/lib/display-prefs";
import {
  dirtyFields,
  groupFields,
  initForm,
  patchBody,
  setValue,
  willFlagStale,
  type FormField,
  type SettingsFormState,
} from "@/lib/settings-form";
import { cn } from "@/lib/utils";

const GROUP_TITLES: Record<string, string> = {
  retrieval: "Retrieval",
  generation: "Generation",
  ingestion: "Ingestion",
};

// Numeric settings that take fractional values (everything else steps by 1).
const FLOAT_SETTINGS = new Set(["SEMANTIC_WEIGHT", "BM25_WEIGHT", "LLM_TEMPERATURE"]);

/**
 * The live settings drawer (spec_v2 §4.7): controls generated from
 * `GET /api/settings` (+ ranges from `GET /api/config`), grouped
 * Retrieval / Generation / Ingestion / Display. Edits stage locally and
 * apply in one PATCH so linked constraints validate as a whole; ingest-time
 * changes warn that they'll mark the corpus stale, and the stale banner
 * carries the "Re-ingest to apply" action.
 */
export function SettingsDrawer() {
  const router = useRouter();
  const { catalog, unreachable, patch } = useSettingsCatalog();
  const [config, setConfig] = useState<ConfigResponse | null>(null);
  const [form, setForm] = useState<SettingsFormState | null>(null);
  const [adopted, setAdopted] = useState<typeof catalog>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    getConfig().then(setConfig, () => undefined);
  }, []);

  // Adopt a newly fetched catalog during render (the React "adjust state
  // when props change" pattern) whenever nothing is staged locally, so a
  // quick-toggle PATCH elsewhere refreshes the drawer too.
  if (catalog !== null && catalog !== adopted) {
    setAdopted(catalog);
    if (form === null || dirtyFields(form).length === 0) {
      setForm(initForm(catalog));
    }
  }

  async function apply() {
    if (form === null || busy) return;
    setBusy(true);
    setError(null);
    try {
      const response = await patch(patchBody(form));
      setForm(initForm(response));
    } catch (failure) {
      setError(failure instanceof ApiError ? failure.message : String(failure));
    } finally {
      setBusy(false);
    }
  }

  async function resetOverride(name: string) {
    setBusy(true);
    setError(null);
    try {
      const response = await patch({ [name]: null });
      setForm(initForm(response));
    } catch (failure) {
      setError(failure instanceof ApiError ? failure.message : String(failure));
    } finally {
      setBusy(false);
    }
  }

  async function reingestNow() {
    try {
      await startIngest(true);
    } catch (failure) {
      setError(failure instanceof ApiError ? failure.message : String(failure));
      return;
    }
    router.push("/corpus");
  }

  const dirty = form !== null ? dirtyFields(form).length : 0;
  const stale = form?.corpusStale ?? false;

  return (
    <Dialog>
      <DialogTrigger
        render={
          <Button variant="ghost" size="sm" aria-label="Settings" className="relative" />
        }
      >
        <SettingsIcon aria-hidden />
        Settings
        {stale && (
          <span
            className="absolute top-0.5 right-0.5 size-2 rounded-full bg-amber-500"
            title="Corpus stale — re-ingest to apply"
          />
        )}
      </DialogTrigger>
      <DialogContent
        className="top-0 right-0 bottom-0 left-auto h-dvh max-h-none w-full translate-x-0 translate-y-0 overflow-y-auto rounded-none sm:max-w-md"
        aria-describedby={undefined}
      >
        <DialogHeader>
          <DialogTitle>Settings</DialogTitle>
          <DialogDescription>
            Query-time knobs apply to the next question. Ingest-time knobs mark
            the corpus stale until a re-ingest.
          </DialogDescription>
        </DialogHeader>

        {unreachable && (
          <p role="alert" className="text-sm text-destructive">
            API unreachable — is the stack up? (docker compose up -d)
          </p>
        )}

        {stale && (
          <div
            role="status"
            className="flex items-center justify-between gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 p-2 text-xs"
          >
            <span>Ingest-time settings changed — the corpus is stale.</span>
            <Button size="xs" variant="outline" onClick={() => void reingestNow()}>
              Re-ingest to apply
            </Button>
          </div>
        )}

        {form === null ? (
          !unreachable && (
            <p className="animate-pulse text-sm text-muted-foreground">Loading…</p>
          )
        ) : (
          <div className="flex flex-col gap-4">
            {groupFields(form).map(([group, fields]) => (
              <fieldset key={group} className="flex flex-col gap-2">
                <legend className="mb-1 text-xs font-semibold tracking-wide text-muted-foreground uppercase">
                  {GROUP_TITLES[group] ?? group}
                </legend>
                {fields.map((field) => (
                  <SettingControl
                    key={field.name}
                    field={field}
                    config={config}
                    disabled={busy}
                    onChange={(value) => setForm(setValue(form, field.name, value))}
                    onReset={() => void resetOverride(field.name)}
                  />
                ))}
              </fieldset>
            ))}

            <DisplaySection />

            {error && (
              <p role="alert" className="text-sm text-destructive">
                {error}
              </p>
            )}

            {form !== null && willFlagStale(form) && (
              <p className="text-xs text-amber-600 dark:text-amber-400">
                Applying these changes marks the corpus stale (content hashes
                don&apos;t change) — re-ingest afterwards to make them real.
              </p>
            )}

            <div className="flex items-center justify-end gap-2 border-t border-border pt-3">
              <span className="mr-auto text-xs text-muted-foreground">
                {dirty > 0 ? `${dirty} unapplied change(s)` : "No staged changes"}
              </span>
              <Button
                variant="outline"
                size="sm"
                disabled={dirty === 0 || busy}
                onClick={() => catalog !== null && setForm(initForm(catalog))}
              >
                Discard
              </Button>
              <Button size="sm" disabled={dirty === 0 || busy} onClick={() => void apply()}>
                {busy ? "Applying…" : "Apply"}
              </Button>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}

/** One generated control: select for choices, checkbox for booleans, number/text otherwise. */
function SettingControl({
  field,
  config,
  disabled,
  onChange,
  onReset,
}: {
  field: FormField;
  config: ConfigResponse | null;
  disabled: boolean;
  onChange: (value: boolean | number | string) => void;
  onReset: () => void;
}) {
  const range = config?.ranges[field.name.toLowerCase()];
  const inputId = `setting-${field.name}`;
  const isBoolean = typeof field.initial === "boolean";
  const isNumber = typeof field.initial === "number";

  return (
    <div className="flex items-center justify-between gap-3">
      <label htmlFor={inputId} className="flex min-w-0 flex-col">
        <span className="truncate font-mono text-xs">{field.name}</span>
        {field.reingestAffecting && (
          <span className="text-[10px] text-muted-foreground">ingest-time</span>
        )}
      </label>
      <div className="flex shrink-0 items-center gap-1">
        {field.choices !== null ? (
          <select
            id={inputId}
            value={String(field.value)}
            disabled={disabled}
            onChange={(event) => onChange(event.target.value)}
            className="h-7 rounded-md border border-border bg-background px-1.5 text-xs"
          >
            {field.choices.map((choice) => (
              <option key={choice} value={choice}>
                {choice}
              </option>
            ))}
          </select>
        ) : isBoolean ? (
          <input
            id={inputId}
            type="checkbox"
            checked={Boolean(field.value)}
            disabled={disabled}
            onChange={(event) => onChange(event.target.checked)}
            className="size-4"
          />
        ) : isNumber ? (
          <input
            id={inputId}
            type="number"
            value={String(field.value)}
            min={range?.min ?? undefined}
            max={range?.max ?? undefined}
            step={FLOAT_SETTINGS.has(field.name) ? 0.05 : 1}
            disabled={disabled}
            onChange={(event) => {
              const parsed = Number(event.target.value);
              if (!Number.isNaN(parsed)) onChange(parsed);
            }}
            className="h-7 w-24 rounded-md border border-border bg-background px-1.5 text-xs tabular-nums"
          />
        ) : (
          <input
            id={inputId}
            type="text"
            value={String(field.value)}
            disabled={disabled}
            onChange={(event) => onChange(event.target.value)}
            className="h-7 w-40 rounded-md border border-border bg-background px-1.5 text-xs"
          />
        )}
        <Button
          variant="ghost"
          size="icon-xs"
          aria-label={`Reset ${field.name} to its env value`}
          className={cn(!field.overridden && "invisible")}
          disabled={disabled}
          onClick={onReset}
        >
          <RotateCcwIcon />
        </Button>
      </div>
    </div>
  );
}

// A constant "am I hydrated?" external store: the client snapshot is
// always true, the server snapshot false — the SSR-safe mounted check.
const subscribeNever = () => () => {};

/** The client-side Display group: theme + reasoning-trace default. */
function DisplaySection() {
  const { theme, setTheme } = useTheme();
  const reasoningOpen = useSyncExternalStore(
    subscribeDisplayPrefs,
    reasoningDefaultOpen,
    reasoningDefaultOpenServer,
  );
  const mounted = useSyncExternalStore(
    subscribeNever,
    () => true,
    () => false,
  );

  return (
    <fieldset className="flex flex-col gap-2">
      <legend className="mb-1 text-xs font-semibold tracking-wide text-muted-foreground uppercase">
        Display
      </legend>
      <div className="flex items-center justify-between gap-3">
        <label htmlFor="setting-theme" className="font-mono text-xs">
          theme
        </label>
        <select
          id="setting-theme"
          value={mounted ? (theme ?? "system") : "system"}
          onChange={(event) => setTheme(event.target.value)}
          className="h-7 rounded-md border border-border bg-background px-1.5 text-xs"
        >
          <option value="light">light</option>
          <option value="dark">dark</option>
          <option value="system">system</option>
        </select>
      </div>
      <div className="flex items-center justify-between gap-3">
        <label htmlFor="setting-reasoning-open" className="font-mono text-xs">
          reasoning trace open by default
        </label>
        <input
          id="setting-reasoning-open"
          type="checkbox"
          checked={reasoningOpen}
          onChange={(event) => setReasoningDefaultOpen(event.target.checked)}
          className="size-4"
        />
      </div>
    </fieldset>
  );
}
