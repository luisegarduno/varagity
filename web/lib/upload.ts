/**
 * Client-side upload validation, mirroring the server's rules
 * (`varagity/api/routes/documents.py`) so obviously-bad files are refused
 * before any bytes leave the browser. The server remains authoritative —
 * its per-file outcomes are still rendered.
 */

export type UploadRejection =
  | "extension_not_allowed"
  | "file_too_large"
  | "invalid_filename";

export interface UploadCheck {
  /** The sanitized basename the server would store under. */
  fileName: string;
  ok: boolean;
  reason?: UploadRejection;
}

/** Reduce a client-supplied name to the basename the server would keep. */
export function safeFileName(raw: string): string | null {
  const name = raw.replace(/\\/g, "/").split("/").pop()?.trim() ?? "";
  if (!name || name.startsWith(".") || name === "..") return null;
  return name;
}

/** The lowercased extension (with dot), or `""` when there is none. */
export function fileExtension(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot <= 0 ? "" : name.slice(dot).toLowerCase();
}

/**
 * Validate one candidate upload against the server's constraints
 * (from `GET /api/config`: `allowed_extensions`, `upload_max_mb`).
 */
export function validateUpload(
  rawName: string,
  sizeBytes: number,
  allowedExtensions: readonly string[],
  maxMb: number,
): UploadCheck {
  const fileName = safeFileName(rawName);
  if (fileName === null) {
    return { fileName: rawName || "(unnamed)", ok: false, reason: "invalid_filename" };
  }
  const allowed = new Set(allowedExtensions.map((ext) => ext.toLowerCase()));
  if (!allowed.has(fileExtension(fileName))) {
    return { fileName, ok: false, reason: "extension_not_allowed" };
  }
  if (sizeBytes > maxMb * 1024 * 1024) {
    return { fileName, ok: false, reason: "file_too_large" };
  }
  return { fileName, ok: true };
}

// ── Relative paths (folder uploads, spec_v3 §5.2) ───────────────────────

// Mirrors of the server's structural bounds (`_MAX_PATH_CHARS`,
// `_MAX_SEGMENT_CHARS`). Depth (`UPLOAD_MAX_PATH_DEPTH`) is a server
// setting the client doesn't know — the server rejects per file with
// `path_too_deep`.
const MAX_PATH_CHARS = 1024;
const MAX_SEGMENT_CHARS = 255;

const PERCENT_ESCAPE = /%[0-9a-fA-F]{2}/;

const RESERVED_SEGMENTS = new Set([
  "CON",
  "PRN",
  "AUX",
  "NUL",
  ...Array.from({ length: 9 }, (_, i) => `COM${i + 1}`),
  ...Array.from({ length: 9 }, (_, i) => `LPT${i + 1}`),
]);

/**
 * Mirror of the server's `_safe_relative_path` (rules 1–3 + shape; the
 * server stays authoritative — depth and the `resolve()` containment
 * backstop live there). Returns the normalized relative path, or `null`
 * when anything is off: absolute paths, `..`, dot-segments (`.git/`,
 * dotfiles), empty segments, control characters, percent-hex escapes,
 * separator lookalikes (NFKC), Windows reserved names, silly lengths.
 */
export function safeRelativePath(raw: string): string | null {
  if (!raw || raw.length > MAX_PATH_CHARS) return null;
  for (const char of raw) {
    const code = char.codePointAt(0) ?? 0;
    if (code < 32 || code === 127) return null;
  }
  const text = raw.replace(/\\/g, "/");
  if (text.startsWith("/")) return null;
  const segments = text.split("/");
  for (const segment of segments) {
    if (!segment || segment.length > MAX_SEGMENT_CHARS) return null;
    if (segment !== segment.trim() || segment.startsWith(".")) return null;
    if (PERCENT_ESCAPE.test(segment)) return null;
    const folded = segment.normalize("NFKC");
    if (folded.includes("/") || folded.includes("\\") || folded.includes(":")) return null;
    if (RESERVED_SEGMENTS.has(segment.split(".")[0].toUpperCase())) return null;
  }
  return segments.join("/");
}

/** Whether any path segment is dot-prefixed (`.git/`, `.DS_Store`, …). */
export function isHiddenPath(raw: string): boolean {
  return raw
    .replace(/\\/g, "/")
    .split("/")
    .some((segment) => segment.trim().startsWith("."));
}

/** Display buckets the attach flow summarizes skipped files into. */
export type SkipLabel =
  | "hidden"
  | "unsupported type"
  | "too large"
  | "invalid path"
  | "folder too deep"
  | "rejected";

/** Map a client- or server-side rejection reason onto its display bucket. */
export function skipLabel(reason: string): SkipLabel {
  switch (reason) {
    case "hidden":
      return "hidden";
    case "extension_not_allowed":
      return "unsupported type";
    case "file_too_large":
      return "too large";
    case "invalid_filename":
    case "invalid_path":
      return "invalid path";
    case "path_too_deep":
      return "folder too deep";
    default:
      return "rejected";
  }
}

/** Display bucket → count of files filtered out. */
export type SkipCounts = Partial<Record<SkipLabel, number>>;

export interface AttachmentPlan {
  /** Files worth uploading. */
  accepted: File[];
  /** One relative path per accepted file; `null` ⇒ flat upload (no paths field). */
  paths: string[] | null;
  /** Display bucket → count of files filtered out client-side. */
  skipped: SkipCounts;
}

/** Bump one bucket of a skip tally. */
export function countSkip(skipped: SkipCounts, label: SkipLabel): void {
  skipped[label] = (skipped[label] ?? 0) + 1;
}

/**
 * Filter a picked file set down to what's worth uploading (spec_v3 §5.3).
 *
 * Client-side filtering is required here, not defense-in-depth: a folder
 * pick ignores `accept`, handing back `.DS_Store`, `.git/`, images — those
 * are counted per bucket for a one-line summary (a 400-file folder must
 * not become 380 rejection rows), never enumerated. A `null`
 * `allowedExtensions` (config not loaded yet) skips the extension/size
 * checks — the server stays authoritative.
 */
export function planAttachments(
  files: readonly File[],
  allowedExtensions: readonly string[] | null,
  maxMb: number | null,
  options: { folder: boolean },
): AttachmentPlan {
  const accepted: File[] = [];
  const paths: string[] = [];
  const skipped: SkipCounts = {};
  for (const file of files) {
    const rawPath = options.folder ? file.webkitRelativePath || file.name : file.name;
    if (options.folder && isHiddenPath(rawPath)) {
      countSkip(skipped, "hidden");
      continue;
    }
    if (allowedExtensions !== null) {
      const check = validateUpload(rawPath, file.size, allowedExtensions, maxMb ?? Infinity);
      if (!check.ok) {
        countSkip(skipped, skipLabel(check.reason ?? "rejected"));
        continue;
      }
    }
    if (options.folder) {
      const relative = safeRelativePath(rawPath);
      if (relative === null) {
        countSkip(skipped, "invalid path");
        continue;
      }
      paths.push(relative);
    }
    accepted.push(file);
  }
  return { accepted, paths: options.folder ? paths : null, skipped };
}

/**
 * One line for the chip: `"312 files skipped — unsupported type"`, or the
 * per-bucket breakdown when several buckets filled. `null` when nothing
 * was skipped.
 */
export function summarizeSkipped(skipped: SkipCounts): string | null {
  const entries = Object.entries(skipped).filter(([, count]) => (count ?? 0) > 0) as [
    SkipLabel,
    number,
  ][];
  const total = entries.reduce((sum, [, count]) => sum + count, 0);
  if (total === 0) return null;
  const noun = total === 1 ? "file" : "files";
  if (entries.length === 1) return `${total} ${noun} skipped — ${entries[0][0]}`;
  const detail = entries.map(([label, count]) => `${count} ${label}`).join(", ");
  return `${total} ${noun} skipped — ${detail}`;
}
