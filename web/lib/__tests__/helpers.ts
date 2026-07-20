// Shared, non-test-suffixed helpers for the upload suites (vitest collects
// only *.test.ts, so this file is imported, never run as a suite).

/** Build a File with an optional folder-pick relative path. */
export function pickedFile(name: string, path?: string, sizeBytes = 8): File {
  const file = new File([new Uint8Array(sizeBytes)], name);
  Object.defineProperty(file, "webkitRelativePath", { value: path ?? "" });
  return file;
}
