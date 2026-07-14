import { describe, expect, it } from "vitest";

import { fileExtension, safeFileName, validateUpload } from "@/lib/upload";

const ALLOWED = [".pdf", ".txt", ".md", ".docx"];

describe("safeFileName", () => {
  it("keeps a plain basename", () => {
    expect(safeFileName("notes.md")).toBe("notes.md");
  });

  it("strips client path components (both separators)", () => {
    expect(safeFileName("../../evil.txt")).toBe("evil.txt");
    expect(safeFileName("C:\\Users\\me\\report.pdf")).toBe("report.pdf");
    expect(safeFileName("nested/dir/name.md")).toBe("name.md");
  });

  it("rejects empty, dot-only, and dotfile-only names", () => {
    expect(safeFileName("")).toBeNull();
    expect(safeFileName("..")).toBeNull();
    expect(safeFileName(".txt")).toBeNull();
    expect(safeFileName("dir/")).toBeNull();
  });
});

describe("fileExtension", () => {
  it("lowercases and includes the dot", () => {
    expect(fileExtension("Report.PDF")).toBe(".pdf");
  });

  it("is empty for extensionless names", () => {
    expect(fileExtension("README")).toBe("");
  });
});

describe("validateUpload", () => {
  it("accepts an allowed extension under the cap", () => {
    expect(validateUpload("notes.md", 1024, ALLOWED, 1)).toEqual({
      fileName: "notes.md",
      ok: true,
    });
  });

  it("matches extensions case-insensitively (both sides)", () => {
    expect(validateUpload("SCAN.PDF", 10, ALLOWED, 1).ok).toBe(true);
    expect(validateUpload("a.txt", 10, [".TXT"], 1).ok).toBe(true);
  });

  it("rejects a disallowed extension", () => {
    expect(validateUpload("virus.exe", 10, ALLOWED, 1)).toEqual({
      fileName: "virus.exe",
      ok: false,
      reason: "extension_not_allowed",
    });
  });

  it("rejects a file over the MB cap (boundary exact)", () => {
    const cap = 1;
    expect(validateUpload("big.txt", cap * 1024 * 1024, ALLOWED, cap).ok).toBe(true);
    expect(validateUpload("big.txt", cap * 1024 * 1024 + 1, ALLOWED, cap)).toEqual({
      fileName: "big.txt",
      ok: false,
      reason: "file_too_large",
    });
  });

  it("rejects unusable names before looking at anything else", () => {
    expect(validateUpload(".txt", 10, ALLOWED, 1)).toEqual({
      fileName: ".txt",
      ok: false,
      reason: "invalid_filename",
    });
  });

  it("validates against the sanitized basename", () => {
    const check = validateUpload("../../evil.txt", 10, ALLOWED, 1);
    expect(check).toEqual({ fileName: "evil.txt", ok: true });
  });
});
