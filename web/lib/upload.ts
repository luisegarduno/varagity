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
