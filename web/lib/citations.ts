/**
 * Inline `[SOURCE]` citation parsing (spec_v2 §4.6).
 *
 * The answer prompt formats every context block as `[SOURCE]:  <path>`
 * and instructs the model to cite the `[SOURCE]` of facts it uses, so
 * answers carry markers like `[SOURCE]: /docs/kelp.txt` (occasionally
 * `[SOURCE: /docs/kelp.txt]`, wrapped in backticks/parens, or trailed by
 * sentence punctuation). This module extracts them, matches each cited
 * path against the answer's evidence, and rewrites the markers into
 * markdown links (`[kelp.txt](#varagity-cite-0)`) the chat renderer turns
 * into chips.
 *
 * Rewriting *before* markdown parsing is load-bearing, not cosmetic: a
 * line-initial `[SOURCE]: /path` is a CommonMark link-reference
 * *definition* and would otherwise be swallowed entirely by the renderer.
 *
 * The `[SOURCE]: <path>` form has no closing delimiter, so its capture
 * stops at the first whitespace — which truncates filenames containing
 * spaces (`/docs/AI Governance.md` → `/docs/AI`). Those captures are
 * *extended* against the evidence rows: if a known source path (full,
 * `/`-suffix, or basename) continues the text at the capture, the whole
 * path is consumed. Only evidence can end a spaced path unambiguously; a
 * spaced path that matches no evidence stays truncated and surfaces as
 * the grounding warning below.
 *
 * A citation whose path matches no evidence row keeps `chunkIndex: null`
 * — the "cited but not in the retrieved evidence" grounding warning.
 */

/** What citation matching needs from one evidence row. */
export interface CitationSourceRef {
  /** Absolute source path (as fed to the model), or `null`. */
  source: string | null;
  /** Basename of the source file, or `null`. */
  fileName: string | null;
}

/** One parsed citation, in answer order. */
export interface Citation {
  /** Position in the answer (also the `#varagity-cite-<id>` link target). */
  id: number;
  /** The cited path, cleaned of wrapping quotes/punctuation. */
  path: string;
  /** Chip label (the path's basename). */
  label: string;
  /**
   * Index of the matching evidence row (best rank wins), or `null` when
   * the citation references a source not in the retrieved set.
   */
  chunkIndex: number | null;
}

/** The rewritten answer plus its extracted citations. */
export interface AnnotatedAnswer {
  /** The answer with each marker replaced by a `#varagity-cite-N` link. */
  markdown: string;
  /** The citations, ordered by appearance (`id` = array index). */
  citations: Citation[];
}

/** Href prefix of the links {@link annotateCitations} writes. */
export const CITATION_HREF_PREFIX = "#varagity-cite-";

// Two marker shapes: `[SOURCE: <path>]` (path inside the bracket) and
// `[SOURCE]: <path>` (path after; colon optional, spacing tolerant — the
// prompt itself uses two spaces). The trailing capture stays on one line
// and stops at whitespace/`]`/`,`/`;` — {@link extendFromEvidence} then
// stretches it over spaces when a known evidence path continues it.
const MARKER_RE =
  /\[SOURCE:\s*([^\]\n]+?)\s*\]|\[SOURCE\][ \t]*:?[ \t]{0,4}([^\s\],;]*)/gi;

// Opening wrappers a model may put around a path, and their closers.
const WRAPPER_PAIRS: Record<string, string> = {
  "`": "`",
  '"': '"',
  "'": "'",
  "(": ")",
  "<": ">",
  "[": "]",
};

/** A captured path token, split into the path and the text to re-emit. */
interface CleanedPath {
  /** The path, stripped of wrappers and trailing punctuation. */
  path: string;
  /**
   * Trailing characters the capture swallowed but that belong to the
   * prose (sentence punctuation, an unpaired `)`); they must re-emit
   * after the chip or the rewrite silently eats them.
   */
  tail: string;
}

/** Strip wrapping quotes/backticks/brackets and trailing punctuation. */
function cleanPath(raw: string): CleanedPath {
  const token = raw.trim();
  const unwrapped = token.replace(/^[`"'(<[]+/, "");
  const dropped = token.slice(0, token.length - unwrapped.length);
  const path = unwrapped.replace(/[`"'>).\],;:!?]+$/, "").trim();
  let tail = unwrapped.slice(path.length).trimStart();
  // Closers pairing with dropped opening wrappers vanish with them; the
  // rest (e.g. a sentence-ending period) survives.
  for (const opener of dropped) {
    const closer = WRAPPER_PAIRS[opener];
    const at = closer ? tail.indexOf(closer) : -1;
    if (at !== -1) tail = tail.slice(0, at) + tail.slice(at + 1);
  }
  return { path, tail };
}

/**
 * Guard against prose false-positives (`the [SOURCE] says …`): a real
 * cited path contains a `/` or a file-extension dot.
 */
function isPathLike(path: string): boolean {
  return /[/.]/.test(path);
}

function basename(path: string): string {
  return path.split("/").filter(Boolean).pop() ?? path;
}

/** Make a path's basename safe as markdown link text. */
function chipLabel(path: string): string {
  const label = basename(path).replace(/[[\]`]/g, "");
  return label || "source";
}

/**
 * Match one cited path against the evidence rows (best rank wins).
 *
 * Tiers, strictest first: exact source path; the cited path as a `/`-
 * separated suffix of a source (relative citation); basename equality
 * with a row's file name. All comparisons case-insensitive.
 */
export function matchSource(
  path: string,
  refs: readonly CitationSourceRef[],
): number | null {
  const cited = path.toLowerCase();
  const citedBase = basename(cited);

  let hit = refs.findIndex((ref) => ref.source?.toLowerCase() === cited);
  if (hit !== -1) return hit;

  hit = refs.findIndex((ref) =>
    ref.source?.toLowerCase().endsWith(`/${cited}`),
  );
  if (hit !== -1) return hit;

  hit = refs.findIndex((ref) => ref.fileName?.toLowerCase() === citedBase);
  return hit === -1 ? null : hit;
}

/** Every string a row can be cited as: full path, `/`-suffixes, basename. */
function candidatePaths(ref: CitationSourceRef): string[] {
  const candidates: string[] = [];
  if (ref.source) {
    candidates.push(ref.source);
    const parts = ref.source.split("/").filter(Boolean);
    for (let i = 1; i < parts.length; i++) {
      candidates.push(parts.slice(i).join("/"));
    }
  }
  if (ref.fileName) candidates.push(ref.fileName);
  return candidates;
}

/** A trailing-form capture stretched past its whitespace stop. */
interface ExtendedCapture {
  /** The full raw token for {@link cleanPath} (wrappers included). */
  raw: string;
  /** Index in the answer just past the consumed text. */
  end: number;
}

/**
 * Stretch a space-truncated `[SOURCE]: <path>` capture over a filename
 * containing spaces.
 *
 * `token` (the regex capture starting at `start`) stopped at the first
 * whitespace, so `/docs/AI Governance.md` arrives as `/docs/AI`. The rest
 * of the line is compared against every way the evidence rows can be
 * cited ({@link candidatePaths}); the longest candidate that continues
 * the text — case-insensitive, ending at a word boundary — wins. Closers
 * pairing any opening wrappers are consumed with it so `cleanPath` sees a
 * balanced token. Returns `null` when no candidate extends the capture,
 * which keeps the unextended behavior for space-free and unknown paths.
 */
function extendFromEvidence(
  text: string,
  start: number,
  token: string,
  refs: readonly CitationSourceRef[],
): ExtendedCapture | null {
  const newline = text.indexOf("\n", start);
  const lineEnd = newline === -1 ? text.length : newline;
  const openers = /^[`"'(<[]+/.exec(token)?.[0] ?? "";
  const body = text.slice(start + openers.length, lineEnd);
  const bodyLower = body.toLowerCase();
  const captured = token.length - openers.length;
  let matched = "";
  for (const ref of refs) {
    for (const candidate of candidatePaths(ref)) {
      if (candidate.length <= captured || candidate.length <= matched.length)
        continue;
      if (!bodyLower.startsWith(candidate.toLowerCase())) continue;
      // Reject mid-word cuts (`AI Governance.md` inside `…Governance.mdx`).
      const after = body[candidate.length];
      if (after !== undefined && /[\w-]/.test(after)) continue;
      matched = body.slice(0, candidate.length);
    }
  }
  if (!matched) return null;
  let end = start + openers.length + matched.length;
  for (const opener of openers) {
    const closer = WRAPPER_PAIRS[opener];
    if (closer && text[end] === closer) end += 1;
  }
  return { raw: text.slice(start, end), end };
}

/**
 * Extract the answer's `[SOURCE]` markers and rewrite them into chip
 * links, matching each against the evidence rows.
 *
 * Trailing-form captures truncated by a space in the filename are first
 * extended against the evidence ({@link extendFromEvidence}), so
 * `[SOURCE]: /docs/AI Governance.md` chips the whole filename instead of
 * a dangling `/docs/AI`. Markers without a usable path (bare `[SOURCE]`,
 * or a token that isn't path-like) are left untouched. Fenced/inline code
 * is not special-cased — grounded answers don't cite from inside code
 * blocks.
 */
export function annotateCitations(
  text: string,
  refs: readonly CitationSourceRef[],
): AnnotatedAnswer {
  const citations: Citation[] = [];
  let markdown = "";
  let cursor = 0;
  MARKER_RE.lastIndex = 0;
  for (let m = MARKER_RE.exec(text); m !== null; m = MARKER_RE.exec(text)) {
    // RegExpExecArray types captures as `string`, but an unmatched
    // alternative's capture is `undefined` at runtime.
    const bracketed = m[1] as string | undefined;
    const trailing = m[2] as string | undefined;
    let raw = bracketed ?? trailing ?? "";
    let end = m.index + m[0].length;
    if (bracketed === undefined && trailing) {
      const extended = extendFromEvidence(
        text,
        end - trailing.length,
        trailing,
        refs,
      );
      if (extended) {
        raw = extended.raw;
        end = extended.end;
      }
    }
    const { path, tail } = cleanPath(raw);
    if (!path || !isPathLike(path)) continue;
    const citation: Citation = {
      id: citations.length,
      path,
      label: chipLabel(path),
      chunkIndex: matchSource(path, refs),
    };
    citations.push(citation);
    markdown += `${text.slice(cursor, m.index)}[${citation.label}](${CITATION_HREF_PREFIX}${citation.id})${tail}`;
    cursor = end;
    MARKER_RE.lastIndex = end;
  }
  return { markdown: markdown + text.slice(cursor), citations };
}

/** Parse a chip link's citation id (`null` for any other href). */
export function citationIdFromHref(
  href: string | undefined | null,
): number | null {
  if (!href || !href.startsWith(CITATION_HREF_PREFIX)) return null;
  const raw = href.slice(CITATION_HREF_PREFIX.length);
  if (!raw) return null; // Number("") is 0 — don't alias to citation 0
  const id = Number(raw);
  return Number.isInteger(id) && id >= 0 ? id : null;
}
