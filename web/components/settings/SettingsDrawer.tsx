"use client";

import { CheckIcon, RotateCcwIcon, SettingsIcon, XIcon } from "lucide-react";
import { useTheme } from "next-themes";
import { useRouter } from "next/navigation";
import { Fragment, useEffect, useState, useSyncExternalStore } from "react";

import { useSettingsCatalog } from "@/components/settings/use-settings";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { ApiError, getConfig, startIngest, type ConfigResponse } from "@/lib/api";
import {
  ACCENTS,
  DENSITIES,
  accent,
  accentServer,
  density,
  densityServer,
  reasoningDefaultOpen,
  reasoningDefaultOpenServer,
  setAccent,
  setDensity,
  setReasoningDefaultOpen,
  subscribeDisplayPrefs,
  type Accent,
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
import { onOpenSettings } from "@/lib/ui-bus";
import { cn } from "@/lib/utils";

const GROUP_TITLES: Record<string, string> = {
  retrieval: "Retrieval",
  generation: "Generation",
  ingestion: "Ingestion",
};

// Numeric settings that take fractional values (everything else steps by 1).
const FLOAT_SETTINGS = new Set(["SEMANTIC_WEIGHT", "BM25_WEIGHT", "LLM_TEMPERATURE"]);

// The right-side sheet reskin of the centered dialog popup: full-height,
// hairline left border, and a translate slide instead of the scale pop.
const SHEET_CLASSES =
  "top-0 right-0 bottom-0 left-auto flex h-dvh max-h-none w-full max-w-full " +
  "translate-x-0 translate-y-0 flex-col gap-0 rounded-none border-l border-border " +
  "p-0 ring-0 shadow-xl transition-[translate] duration-300 " +
  "data-starting-style:translate-x-full data-starting-style:translate-y-0 " +
  "data-starting-style:scale-100 data-starting-style:opacity-100 " +
  "data-ending-style:translate-x-full data-ending-style:translate-y-0 " +
  "data-ending-style:scale-100 data-ending-style:opacity-100 sm:max-w-md";

/**
 * The live settings drawer (spec_v2 §4.7): controls generated from
 * `GET /api/settings` (+ ranges from `GET /api/config`), grouped
 * Retrieval / Generation / Ingestion / Display. Edits stage locally and
 * apply in one PATCH so linked constraints validate as a whole; ingest-time
 * changes warn that they'll mark the corpus stale, and the stale banner
 * carries the "Re-ingest to apply" action.
 *
 * The dialog is controlled so the ⌘K palette can open it through the UI
 * bus; `openOnBusEvent` lets a secondary mount (the mobile navigation
 * drawer's footer) opt out so a palette open never raises two dialogs.
 */
export function SettingsDrawer({
  openOnBusEvent = true,
}: {
  openOnBusEvent?: boolean;
}) {
  const router = useRouter();
  const { catalog, unreachable, patch } = useSettingsCatalog();
  const [open, setOpen] = useState(false);
  const [config, setConfig] = useState<ConfigResponse | null>(null);
  const [form, setForm] = useState<SettingsFormState | null>(null);
  const [adopted, setAdopted] = useState<typeof catalog>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    getConfig().then(setConfig, () => undefined);
  }, []);

  // The palette (and other global surfaces) ask us to open via the UI bus;
  // state changes only inside the event listener.
  useEffect(() => {
    if (!openOnBusEvent) return;
    return onOpenSettings(() => setOpen(true));
  }, [openOnBusEvent]);

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
    setOpen(false);
    router.push("/corpus");
  }

  const dirty = form !== null ? dirtyFields(form).length : 0;
  const stale = form?.corpusStale ?? false;

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger
        render={
          <Button
            variant="ghost"
            size="sm"
            aria-label="Settings"
            className="relative flex-1 justify-start"
          />
        }
      >
        <SettingsIcon aria-hidden />
        Settings
        {stale && (
          <span
            className="absolute top-1 right-1 size-1.5 rounded-full bg-amber-500"
            title="Corpus stale — re-ingest to apply"
          />
        )}
      </DialogTrigger>
      <DialogContent className={SHEET_CLASSES} showCloseButton={false}>
        <DialogHeader className="border-b border-border px-5 py-4">
          <div className="flex items-start justify-between gap-2">
            <DialogTitle>Settings</DialogTitle>
            <DialogClose
              render={
                <Button
                  variant="ghost"
                  size="icon-sm"
                  aria-label="Close settings"
                  className="-mt-0.5 -mr-1.5"
                />
              }
            >
              <XIcon />
            </DialogClose>
          </div>
          <DialogDescription>
            Query-time knobs apply to the next question. Ingest-time knobs mark
            the corpus stale until a re-ingest.
          </DialogDescription>
        </DialogHeader>

        <div className="flex min-h-0 flex-1 flex-col gap-5 overflow-y-auto px-5 py-4 scroll-fade-y">
          {unreachable && (
            <p role="alert" className="text-sm text-destructive">
              API unreachable — is the stack up? (docker compose up -d)
            </p>
          )}

          {stale && (
            <div
              role="status"
              className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2.5"
            >
              <span className="flex items-center gap-2 text-xs">
                <Badge variant="warning">stale</Badge>
                Ingest-time settings changed.
              </span>
              <Button size="xs" variant="outline" onClick={() => void reingestNow()}>
                Re-ingest to apply
              </Button>
            </div>
          )}

          {form === null ? (
            !unreachable && (
              <p className="text-sm text-muted-foreground motion-safe:animate-pulse">
                Loading…
              </p>
            )
          ) : (
            <>
              {groupFields(form).map(([group, fields]) => (
                <Fragment key={group}>
                  <fieldset className="flex flex-col gap-3">
                    <legend className="mb-2 text-[11px] font-medium tracking-[0.08em] text-muted-foreground uppercase">
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
                  <Separator />
                </Fragment>
              ))}

              <DisplaySection />

              {error && (
                <p role="alert" className="text-sm text-destructive">
                  {error}
                </p>
              )}
            </>
          )}
        </div>

        {form !== null && (
          <div className="flex flex-col gap-2 border-t border-border px-5 py-3">
            {willFlagStale(form) && (
              <p className="text-xs text-amber-600 dark:text-amber-400">
                Applying these changes marks the corpus stale (content hashes
                don&apos;t change) — re-ingest afterwards to make them real.
              </p>
            )}
            <div className="flex items-center gap-2">
              <span className="mr-auto text-xs text-muted-foreground tabular-nums">
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

/** One generated control: Select for choices, Switch for booleans, Input otherwise. */
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
  const dirtyField = field.value !== field.initial;

  return (
    <div className="flex items-center justify-between gap-3">
      <label htmlFor={inputId} className="flex min-w-0 flex-col gap-0.5">
        <span className="flex items-center gap-1.5 font-mono text-xs">
          <span className="truncate">{field.name.toLowerCase()}</span>
          {dirtyField && (
            <span
              className="size-1.5 shrink-0 rounded-full bg-primary"
              title="Staged — not applied yet"
            />
          )}
        </span>
        {field.reingestAffecting && (
          <span className="text-[10px] text-muted-foreground">ingest-time</span>
        )}
      </label>
      <div className="flex shrink-0 items-center gap-1">
        {field.choices !== null ? (
          <Select
            value={String(field.value)}
            onValueChange={(value) => {
              if (typeof value === "string") onChange(value);
            }}
            disabled={disabled}
          >
            <SelectTrigger id={inputId} className="h-7 w-44 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {field.choices.map((choice) => (
                <SelectItem key={choice} value={choice} className="text-xs">
                  {choice}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        ) : isBoolean ? (
          <Switch
            id={inputId}
            checked={Boolean(field.value)}
            disabled={disabled}
            onCheckedChange={(checked) => onChange(checked)}
          />
        ) : isNumber ? (
          <Input
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
            className="h-7 w-24 text-xs tabular-nums md:text-xs"
          />
        ) : (
          <Input
            id={inputId}
            type="text"
            value={String(field.value)}
            disabled={disabled}
            onChange={(event) => onChange(event.target.value)}
            className="h-7 w-40 text-xs md:text-xs"
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

// Swatch fills mirror the globals.css accent contract (hue/chroma per
// accent at the primary lightness) — inline so each swatch keeps its own
// color regardless of the accent currently applied to <html>.
const ACCENT_SWATCHES: Record<Accent, string> = {
  indigo: "oklch(0.5 0.14 278)",
  teal: "oklch(0.5 0.082 195)",
  violet: "oklch(0.5 0.15 303)",
  ember: "oklch(0.5 0.115 50)",
};

const THEME_CHOICES = ["light", "dark", "system"] as const;

/**
 * The client-side Display group: theme, accent, density, and the
 * reasoning-trace default. These never PATCH the server — theme belongs to
 * next-themes, the rest to localStorage display prefs.
 */
function DisplaySection() {
  const { theme, setTheme } = useTheme();
  const reasoningOpen = useSyncExternalStore(
    subscribeDisplayPrefs,
    reasoningDefaultOpen,
    reasoningDefaultOpenServer,
  );
  const currentAccent = useSyncExternalStore(
    subscribeDisplayPrefs,
    accent,
    accentServer,
  );
  const currentDensity = useSyncExternalStore(
    subscribeDisplayPrefs,
    density,
    densityServer,
  );
  const mounted = useSyncExternalStore(
    subscribeNever,
    () => true,
    () => false,
  );

  return (
    <fieldset className="flex flex-col gap-3">
      <legend className="mb-2 text-[11px] font-medium tracking-[0.08em] text-muted-foreground uppercase">
        Display
      </legend>

      <div className="flex items-center justify-between gap-3">
        <label htmlFor="setting-theme" className="font-mono text-xs">
          theme
        </label>
        <Select
          value={mounted ? (theme ?? "system") : "system"}
          onValueChange={(value) => {
            if (typeof value === "string") setTheme(value);
          }}
        >
          <SelectTrigger id="setting-theme" className="h-7 w-44 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {THEME_CHOICES.map((choice) => (
              <SelectItem key={choice} value={choice} className="text-xs">
                {choice}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="flex items-center justify-between gap-3">
        <span id="setting-accent-label" className="font-mono text-xs">
          accent
        </span>
        <div
          role="group"
          aria-labelledby="setting-accent-label"
          className="flex items-center gap-1.5"
        >
          {ACCENTS.map((name) => {
            const active = currentAccent === name;
            return (
              <button
                key={name}
                type="button"
                aria-pressed={active}
                aria-label={`${name} accent`}
                title={name}
                onClick={() => setAccent(name)}
                className={cn(
                  "flex size-6 items-center justify-center rounded-full border border-foreground/15",
                  "motion-safe:transition-shadow motion-safe:duration-150",
                  active && "ring-2 ring-ring ring-offset-2 ring-offset-popover",
                )}
                style={{ backgroundColor: ACCENT_SWATCHES[name] }}
              >
                {active && <CheckIcon className="size-3.5 text-white" aria-hidden />}
              </button>
            );
          })}
        </div>
      </div>

      <div className="flex items-center justify-between gap-3">
        <span id="setting-density-label" className="font-mono text-xs">
          density
        </span>
        <div
          role="group"
          aria-labelledby="setting-density-label"
          className="flex rounded-lg border border-border p-0.5"
        >
          {DENSITIES.map((name) => {
            const active = currentDensity === name;
            return (
              <button
                key={name}
                type="button"
                aria-pressed={active}
                onClick={() => setDensity(name)}
                className={cn(
                  "rounded-md px-2 py-0.5 text-xs capitalize transition-colors",
                  active
                    ? "bg-muted text-foreground"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                {name}
              </button>
            );
          })}
        </div>
      </div>

      <div className="flex items-center justify-between gap-3">
        <label htmlFor="setting-reasoning-open" className="font-mono text-xs">
          reasoning trace open by default
        </label>
        <Switch
          id="setting-reasoning-open"
          checked={reasoningOpen}
          onCheckedChange={(checked) => setReasoningDefaultOpen(checked)}
        />
      </div>
    </fieldset>
  );
}
