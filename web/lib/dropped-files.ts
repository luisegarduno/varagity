/**
 * Turning a drop into files (spec_v3 §5.2).
 *
 * `DataTransfer.files` flattens a dropped folder into one entry for the
 * directory itself — no extension, no contents — so a folder drop reads as
 * an unsupported type. The entries API (`webkitGetAsEntry`) is the only way
 * to see inside one, so folder drops walk it and normalize each leaf into
 * the shape the directory picker already produces: a `File` carrying its
 * `webkitRelativePath`. Both upload surfaces then share one representation
 * and one filter (`planAttachments`).
 */

import { countSkip, type SkipCounts } from "@/lib/upload";

/**
 * A runaway guard, not a policy: real corpora nest a few deep, and the
 * server's `UPLOAD_MAX_PATH_DEPTH` (12 by default) is the real limit — it
 * rejects per file with `path_too_deep`. This only stops a pathological
 * tree from hanging the tab. A pruned subtree counts once, not per file:
 * its contents were never read.
 */
const MAX_WALK_DEPTH = 64;

/**
 * The slice of `FileSystemEntry` the walk needs. Duck-typed on purpose:
 * the concrete types are split across `FileSystemFileEntry` /
 * `FileSystemDirectoryEntry`, and the unit suite's fakes implement this
 * shape without a DOM.
 */
interface EntryLike {
  readonly isFile: boolean;
  readonly isDirectory: boolean;
  readonly name: string;
  file?: (ok: (file: File) => void, fail: (error: unknown) => void) => void;
  createReader?: () => ReaderLike;
}

/** The slice of `FileSystemDirectoryReader` the walk needs. */
interface ReaderLike {
  readEntries: (
    ok: (entries: EntryLike[]) => void,
    fail: (error: unknown) => void,
  ) => void;
}

/** What a drop yielded, ready for `planAttachments`. */
export interface DroppedFiles {
  /** Leaf files, each carrying its `webkitRelativePath` when foldered. */
  files: File[];
  /** True when a directory was dropped — selects the folder upload path. */
  folder: boolean;
  /** Buckets pruned during the walk itself (see `MAX_WALK_DEPTH`). */
  skipped: SkipCounts;
}

/** Promise-wrap `FileSystemFileEntry.file()`; unreadable ⇒ `null`. */
function readFile(entry: EntryLike): Promise<File | null> {
  return new Promise((resolve) => {
    if (!entry.file) {
      resolve(null);
      return;
    }
    entry.file(
      (file) => resolve(file),
      () => resolve(null),
    );
  });
}

/**
 * Drain a directory reader. `readEntries` yields one *batch* per call
 * (Chrome caps it at 100) and signals the end with an empty one — a single
 * call silently truncates every folder past that cap.
 */
async function readAll(reader: ReaderLike): Promise<EntryLike[]> {
  const entries: EntryLike[] = [];
  for (;;) {
    const batch = await new Promise<EntryLike[]>((resolve) => {
      reader.readEntries(
        (next) => resolve(next),
        () => resolve([]),
      );
    });
    if (batch.length === 0) return entries;
    entries.push(...batch);
  }
}

/**
 * Stamp the folder-pick field onto a file the entries API produced.
 * `webkitdirectory` populates `webkitRelativePath`; `FileSystemFileEntry.file()`
 * leaves it `""`. Writing it here is what lets `planAttachments` read one
 * field for both surfaces.
 */
function withRelativePath(file: File, path: string): File {
  Object.defineProperty(file, "webkitRelativePath", {
    value: path,
    configurable: true,
  });
  return file;
}

/** Depth-first walk of one dropped entry, collecting leaf files. */
async function walk(
  entry: EntryLike,
  prefix: string,
  out: File[],
  skipped: SkipCounts,
  depth: number,
): Promise<void> {
  if (entry.isFile) {
    const file = await readFile(entry);
    if (file !== null) out.push(withRelativePath(file, `${prefix}${entry.name}`));
    return;
  }
  if (!entry.isDirectory || !entry.createReader) return;
  if (depth >= MAX_WALK_DEPTH) {
    countSkip(skipped, "folder too deep");
    return;
  }
  for (const child of await readAll(entry.createReader())) {
    await walk(child, `${prefix}${entry.name}/`, out, skipped, depth + 1);
  }
}

/**
 * Read a drop into uploadable files, descending into any dropped folders.
 *
 * Args:
 *   transfer: The drop event's `dataTransfer`. Read synchronously up to the
 *     first await — see below — so this is safe to call unawaited from a
 *     `onDrop` handler.
 *
 * Returns:
 *   The leaf files, whether a directory was involved, and anything the walk
 *   itself pruned. A drop with no directory returns `dataTransfer.files`
 *   verbatim, keeping the flat path byte-identical.
 */
export async function filesFromDrop(transfer: DataTransfer): Promise<DroppedFiles> {
  // `items` and `files` are only alive during the drop event's dispatch —
  // the first await neuters them. Take every handle up front, walk after.
  const flat = Array.from(transfer.files ?? []);
  const roots: EntryLike[] = [];
  let folder = false;
  for (const item of Array.from(transfer.items ?? [])) {
    if (item.kind !== "file") continue;
    const entry = item.webkitGetAsEntry?.() as EntryLike | null;
    if (!entry) continue;
    roots.push(entry);
    if (entry.isDirectory) folder = true;
  }
  // No directory in the drop ⇒ `files` already says everything, and the
  // flat contract stays exactly as it was (no paths field, no hidden
  // filter). This is also the fallback when the entries API is missing.
  if (!folder) return { files: flat, folder: false, skipped: {} };

  const files: File[] = [];
  const skipped: SkipCounts = {};
  for (const entry of roots) await walk(entry, "", files, skipped, 1);
  return { files, folder: true, skipped };
}
