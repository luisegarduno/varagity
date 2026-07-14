/**
 * Pure state for the settings drawer (spec_v2 §4.7).
 *
 * The drawer stages edits locally and applies them in one PATCH so linked
 * constraints (the fusion weight pair) validate as a whole; these helpers
 * own the staging, dirtiness, the weight linkage, and the "will this
 * flag the corpus stale?" signal the UI warns with before applying.
 */
import type { SettingsResponse, SettingValue } from "@/lib/api";

/** One drawer control, staged value included. */
export interface FormField {
  name: string;
  group: string;
  reingestAffecting: boolean;
  overridden: boolean;
  choices: string[] | null;
  /** The server's effective value (what "unchanged" means). */
  initial: SettingValue;
  /** The staged value shown in the control. */
  value: SettingValue;
}

export interface SettingsFormState {
  fields: FormField[];
  corpusStale: boolean;
}

/** The linked sum-to-1 fusion pair (config validator: the linked slider). */
const WEIGHT_PAIR: Record<string, string> = {
  SEMANTIC_WEIGHT: "BM25_WEIGHT",
  BM25_WEIGHT: "SEMANTIC_WEIGHT",
};

/** Build the form from a `GET`/`PATCH /api/settings` response. */
export function initForm(catalog: SettingsResponse): SettingsFormState {
  return {
    fields: catalog.settings.map((setting) => ({
      name: setting.name,
      group: setting.group,
      reingestAffecting: setting.reingest_affecting,
      overridden: setting.overridden,
      choices: setting.choices ?? null,
      initial: setting.value,
      value: setting.value,
    })),
    corpusStale: catalog.corpus_stale,
  };
}

/**
 * Stage one edit. Editing either fusion weight sets its partner to the
 * complement, so the pair always sums to 1.0 and a lone-weight 422 can't
 * happen from the drawer.
 */
export function setValue(
  state: SettingsFormState,
  name: string,
  value: SettingValue,
): SettingsFormState {
  const partner = WEIGHT_PAIR[name];
  const complement =
    partner !== undefined && typeof value === "number"
      ? Math.round((1 - value) * 1000) / 1000
      : null;
  return {
    ...state,
    fields: state.fields.map((field) => {
      if (field.name === name) return { ...field, value };
      if (partner !== undefined && field.name === partner && complement !== null) {
        return { ...field, value: complement };
      }
      return field;
    }),
  };
}

/** The staged-but-unapplied fields. */
export function dirtyFields(state: SettingsFormState): FormField[] {
  return state.fields.filter((field) => field.value !== field.initial);
}

/** The `PATCH /api/settings` overrides body for the staged edits. */
export function patchBody(state: SettingsFormState): Record<string, SettingValue> {
  return Object.fromEntries(dirtyFields(state).map((field) => [field.name, field.value]));
}

/**
 * Whether applying the staged edits will flag the corpus stale — an
 * ingest-time knob changed (it won't change content hashes; the corpus
 * needs a re-ingest to actually reflect it).
 */
export function willFlagStale(state: SettingsFormState): boolean {
  return dirtyFields(state).some((field) => field.reingestAffecting);
}

/** Group fields for rendering, in the spec §4.7 drawer order. */
export function groupFields(state: SettingsFormState): [string, FormField[]][] {
  const order = ["retrieval", "generation", "ingestion"];
  return order
    .map((group): [string, FormField[]] => [
      group,
      state.fields.filter((field) => field.group === group),
    ])
    .filter(([, fields]) => fields.length > 0);
}
