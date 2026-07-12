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
// and stops at whitespace/`]`/`,`/`;`.
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

/**
 * Extract the answer's `[SOURCE]` markers and rewrite them into chip
 * links, matching each against the evidence rows.
 *
 * Markers without a usable path (bare `[SOURCE]`, or a token that isn't
 * path-like) are left untouched. Fenced/inline code is not special-cased
 * — grounded answers don't cite from inside code blocks.
 */
export function annotateCitations(
  text: string,
  refs: readonly CitationSourceRef[],
): AnnotatedAnswer {
  const citations: Citation[] = [];
  const markdown = text.replace(
    MARKER_RE,
    (full: string, bracketed?: string, trailing?: string) => {
      const { path, tail } = cleanPath(bracketed ?? trailing ?? "");
      if (!path || !isPathLike(path)) return full;
      const citation: Citation = {
        id: citations.length,
        path,
        label: chipLabel(path),
        chunkIndex: matchSource(path, refs),
      };
      citations.push(citation);
      return `[${citation.label}](${CITATION_HREF_PREFIX}${citation.id})${tail}`;
    },
  );
  return { markdown, citations };
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
