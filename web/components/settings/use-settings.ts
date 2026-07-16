"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback } from "react";

import {
  patchSettings,
  type SettingsResponse,
  type SettingValue,
} from "@/lib/api";
import { settingsQuery } from "@/lib/queries";
import { notifySettingsChanged } from "@/lib/settings-bus";

/**
 * Shared access to the runtime settings catalog (spec_v2 §4.7). Every
 * consumer (drawer, composer quick-toggles, stale indicators) reads the
 * same cache entry, so a PATCH from any of them repaints all of them at
 * once — no per-consumer subscription, no per-consumer refetch.
 */
export function useSettingsCatalog() {
  const queryClient = useQueryClient();
  const { data: catalog = null, isError: unreachable } =
    useQuery(settingsQuery());

  const patch = useCallback(
    async (
      overrides: Record<string, SettingValue | null>,
    ): Promise<SettingsResponse> => {
      const response = await patchSettings(overrides);
      // The response *is* the new catalog, so paint it without a round
      // trip; the bus then has every surface confirm against the server
      // (which is also how an out-of-band change, like a re-ingest
      // clearing the stale flag, gets picked up).
      queryClient.setQueryData(settingsQuery().queryKey, response);
      notifySettingsChanged();
      return response;
    },
    [queryClient],
  );

  return { catalog, unreachable, patch };
}

/** Pick one setting's effective value out of the catalog. */
export function settingValue(
  catalog: SettingsResponse | null,
  name: string,
): SettingValue | null {
  return catalog?.settings.find((setting) => setting.name === name)?.value ?? null;
}

/** Pick one setting's choices out of the catalog. */
export function settingChoices(catalog: SettingsResponse | null, name: string): string[] {
  return catalog?.settings.find((setting) => setting.name === name)?.choices ?? [];
}
