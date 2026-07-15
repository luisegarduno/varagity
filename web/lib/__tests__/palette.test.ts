import { describe, expect, it } from "vitest";

import {
  filterCommands,
  groupCommands,
  type PaletteCommand,
} from "@/lib/palette";

function command(
  partial: Pick<PaletteCommand, "id" | "label"> & Partial<PaletteCommand>,
): PaletteCommand {
  return { group: "Actions", ...partial };
}

const NEW_CHAT = command({
  id: "action:new-chat",
  label: "New chat",
  keywords: ["create", "conversation"],
});
const FOCUS_COMPOSER = command({
  id: "action:focus-composer",
  label: "Focus composer",
  keywords: ["input", "message"],
});
const CORPUS = command({
  id: "navigate:corpus",
  label: "Corpus",
  group: "Navigate",
  keywords: ["documents", "files"],
});
const SETTINGS = command({
  id: "navigate:settings",
  label: "Settings",
  group: "Navigate",
  keywords: ["preferences"],
});
const THEME_DARK = command({
  id: "theme:dark",
  label: "Theme: Dark",
  group: "Appearance",
  keywords: ["appearance", "night"],
});

const ALL = [NEW_CHAT, FOCUS_COMPOSER, CORPUS, SETTINGS, THEME_DARK];

function ids(commands: PaletteCommand[]): string[] {
  return commands.map((entry) => entry.id);
}

describe("filterCommands", () => {
  it("returns every command in original order for an empty query", () => {
    expect(filterCommands(ALL, "")).toEqual(ALL);
  });

  it("treats a whitespace-only query as empty", () => {
    expect(filterCommands(ALL, "   ")).toEqual(ALL);
  });

  it("does not mutate the input list", () => {
    const input = [...ALL];
    filterCommands(input, "corpus");
    expect(input).toEqual(ALL);
  });

  it("ranks a label prefix above a label substring", () => {
    const openSettings = command({ id: "a", label: "Open settings" });
    const settings = command({ id: "b", label: "Settings" });
    // Original order has the substring match first; the prefix match wins.
    expect(ids(filterCommands([openSettings, settings], "settings"))).toEqual([
      "b",
      "a",
    ]);
  });

  it("matches on keywords, ranked below any label match", () => {
    // "co": label prefix on Corpus, label substring in "composer",
    // keyword-only via "conversation" on New chat.
    expect(ids(filterCommands(ALL, "co"))).toEqual([
      "navigate:corpus",
      "action:focus-composer",
      "action:new-chat",
    ]);
  });

  it("matches keyword word-prefixes and substrings", () => {
    expect(ids(filterCommands(ALL, "pref"))).toEqual(["navigate:settings"]);
    expect(ids(filterCommands(ALL, "igh"))).toEqual(["theme:dark"]); // n[igh]t
  });

  it("is case-insensitive in both the query and the command", () => {
    expect(ids(filterCommands(ALL, "THEME"))).toEqual(["theme:dark"]);
    expect(ids(filterCommands(ALL, "nEw ChAt"))).toEqual(["action:new-chat"]);
  });

  it("trims the query before matching", () => {
    expect(ids(filterCommands(ALL, "  corpus  "))).toEqual(["navigate:corpus"]);
  });

  it("keeps the original order within a rank", () => {
    const first = command({ id: "one", label: "Alpha evidence" });
    const second = command({ id: "two", label: "Beta evidence" });
    expect(ids(filterCommands([first, second], "evidence"))).toEqual([
      "one",
      "two",
    ]);
    expect(ids(filterCommands([second, first], "evidence"))).toEqual([
      "two",
      "one",
    ]);
  });

  it("returns an empty list when nothing matches", () => {
    expect(filterCommands(ALL, "zzz-no-such-command")).toEqual([]);
  });
});

describe("groupCommands", () => {
  it("groups in first-seen order, keeping item order within groups", () => {
    const groups = groupCommands([
      NEW_CHAT,
      CORPUS,
      FOCUS_COMPOSER,
      THEME_DARK,
      SETTINGS,
    ]);
    expect(groups.map((group) => group.group)).toEqual([
      "Actions",
      "Navigate",
      "Appearance",
    ]);
    expect(ids(groups[0].items)).toEqual([
      "action:new-chat",
      "action:focus-composer",
    ]);
    expect(ids(groups[1].items)).toEqual([
      "navigate:corpus",
      "navigate:settings",
    ]);
    expect(ids(groups[2].items)).toEqual(["theme:dark"]);
  });

  it("follows a ranked list's group order (regroup after filtering)", () => {
    // Ranking puts the Navigate match first; its group leads the display.
    const ranked = filterCommands(ALL, "co");
    const groups = groupCommands(ranked);
    expect(groups.map((group) => group.group)).toEqual([
      "Navigate",
      "Actions",
    ]);
  });

  it("returns an empty list for no commands", () => {
    expect(groupCommands([])).toEqual([]);
  });
});
