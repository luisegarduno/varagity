"use client";

import { useCallback, useEffect, useState } from "react";

import {
  getSettings,
  patchSettings,
  type SettingsResponse,
  type SettingValue,
} from "@/lib/api";
import { notifySettingsChanged, onSettingsChanged } from "@/lib/settings-bus";

/**
 * Shared access to the runtime settings catalog (spec_v2 §4.7). Every
 * consumer (drawer, composer quick-toggles, stale indicators) sees the
 * same override layer: a PATCH from any of them re-broadcasts on the
 * settings bus, and everyone refetches.
 */
export function useSettingsCatalog() {
  const [catalog, setCatalog] = useState<SettingsResponse | null>(null);
  const [unreachable, setUnreachable] = useState(false);

  const refresh = useCallback(() => {
    getSettings().then(
      (response) => {
        setCatalog(response);
        setUnreachable(false);
      },
      () => setUnreachable(true),
    );
  }, []);

  useEffect(() => {
    refresh();
    return onSettingsChanged(refresh);
  }, [refresh]);

  const patch = useCallback(
    async (overrides: Record<string, SettingValue | null>): Promise<SettingsResponse> => {
      const response = await patchSettings(overrides);
      setCatalog(response);
      notifySettingsChanged();
      return response;
    },
    [],
  );

  return { catalog, unreachable, refresh, patch };
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
