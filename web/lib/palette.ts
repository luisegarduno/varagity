/**
 * Pure command-palette logic (Phase 9): the serializable command shape plus
 * the filter/rank and regroup helpers the palette component renders from.
 * Deliberately React-free and handler-free — commands carry only data, and
 * the component maps command ids onto the actions they trigger.
 */

/**
 * One palette entry. Serializable on purpose (no functions): `id` is the
 * dispatch key the component switches on, `hint` a right-aligned annotation.
 */
export interface PaletteCommand {
  id: string;
  label: string;
  group: string;
  keywords?: string[];
  hint?: string;
}

/** One display group of commands, as produced by {@link groupCommands}. */
export interface PaletteGroup<T extends PaletteCommand = PaletteCommand> {
  group: string;
  items: T[];
}

// Match ranks, lower sorts first. A prefix hit is also a substring hit, so
// the checks run best-first and stop at the first that lands.
const RANK_LABEL_PREFIX = 0;
const RANK_LABEL_SUBSTRING = 1;
const RANK_KEYWORD = 2;

/**
 * Best match rank for one command against a normalized query, or `null`
 * when it does not match. `includes` covers both word-prefix and substring
 * matches; keywords rank below any label match.
 */
function rankFor(command: PaletteCommand, query: string): number | null {
  const label = command.label.toLowerCase();
  if (label.startsWith(query)) return RANK_LABEL_PREFIX;
  if (label.includes(query)) return RANK_LABEL_SUBSTRING;
  for (const keyword of command.keywords ?? []) {
    if (keyword.toLowerCase().includes(query)) return RANK_KEYWORD;
  }
  return null;
}

/**
 * Filter and rank commands against a free-text query.
 *
 * The query is trimmed and matched case-insensitively; an empty (or
 * whitespace-only) query returns every command in its original order.
 * Matching considers the label and the keywords, counting both word-prefix
 * and substring hits. Ranking: label prefix > label substring > keyword
 * match, stable (original order) within each rank. Returns a flat ranked
 * list — callers regroup for display via {@link groupCommands}.
 */
export function filterCommands<T extends PaletteCommand>(
  commands: T[],
  query: string,
): T[] {
  const needle = query.trim().toLowerCase();
  if (needle === "") return [...commands];

  const ranked: { command: T; rank: number; index: number }[] = [];
  commands.forEach((command, index) => {
    const rank = rankFor(command, needle);
    if (rank !== null) ranked.push({ command, rank, index });
  });
  ranked.sort((a, b) => a.rank - b.rank || a.index - b.index);
  return ranked.map((entry) => entry.command);
}

/**
 * Regroup a flat command list for display, preserving the order in which
 * each group is first seen and the given item order within each group.
 */
export function groupCommands<T extends PaletteCommand>(
  commands: T[],
): PaletteGroup<T>[] {
  const groups: PaletteGroup<T>[] = [];
  const byName = new Map<string, PaletteGroup<T>>();
  for (const command of commands) {
    let bucket = byName.get(command.group);
    if (!bucket) {
      bucket = { group: command.group, items: [] };
      byName.set(command.group, bucket);
      groups.push(bucket);
    }
    bucket.items.push(command);
  }
  return groups;
}
