import { describe, expect, it } from "vitest";

import {
  fileExtension,
  isHiddenPath,
  planAttachments,
  safeFileName,
  safeRelativePath,
  skipLabel,
  summarizeSkipped,
  validateUpload,
} from "@/lib/upload";

import { pickedFile } from "./helpers";

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

describe("safeRelativePath", () => {
  it.each([
    "../evil.md",
    "..\\evil.md",
    "q3/../../evil.md",
    "/etc/evil.md",
    "C:\\evil.md",
    "C:/evil.md",
    "%2e%2e/evil.md",
    "q3/%2fevil.md",
    "q3／evil.md", // fullwidth solidus — NFKC-normalizes to "/"
    ".git/config",
    "q3/.hidden.md",
    "q3//evil.md",
    "q3/./evil.md",
    "q3/evil.md/",
    "q3/ evil.md",
    "CON/evil.md",
    "q3/NUL.md",
    "",
    "a/".repeat(600) + "evil.md",
    "q3/" + "x".repeat(300) + ".md",
  ])("rejects %j (the server's traversal table, mirrored)", (hostile) => {
    expect(safeRelativePath(hostile)).toBeNull();
  });

  it("rejects control characters and NULs", () => {
    expect(safeRelativePath("q3/ev\x00il.md")).toBeNull();
    expect(safeRelativePath("q3/ev\x07il.md")).toBeNull();
  });

  it.each([
    ["notes.md", "notes.md"],
    ["q3/notes.md", "q3/notes.md"],
    ["reports/2026/q3/notes.md", "reports/2026/q3/notes.md"],
    ["q3\\notes.md", "q3/notes.md"],
    ["Ünïcode/nötes.md", "Ünïcode/nötes.md"],
  ])("normalizes contained path %j to %j", (raw, expected) => {
    expect(safeRelativePath(raw)).toBe(expected);
  });
});

describe("isHiddenPath", () => {
  it("flags any dot-prefixed segment", () => {
    expect(isHiddenPath("repo/.git/config")).toBe(true);
    expect(isHiddenPath(".DS_Store")).toBe(true);
    expect(isHiddenPath("docs/.obsidian/app.json")).toBe(true);
  });

  it("passes visible paths", () => {
    expect(isHiddenPath("docs/notes.md")).toBe(false);
    expect(isHiddenPath("v1.2/notes.md")).toBe(false);
  });
});

describe("planAttachments", () => {
  it("keeps a folder pick's structure, positionally aligned", () => {
    const plan = planAttachments(
      [pickedFile("a.md", "corpus/q3/a.md"), pickedFile("b.md", "corpus/q4/b.md")],
      ALLOWED,
      1,
      { folder: true },
    );
    expect(plan.accepted.map((file) => file.name)).toEqual(["a.md", "b.md"]);
    expect(plan.paths).toEqual(["corpus/q3/a.md", "corpus/q4/b.md"]);
    expect(plan.skipped).toEqual({});
  });

  it("summarizes a messy folder instead of enumerating it (400 → 88 kept)", () => {
    const files = [
      ...Array.from({ length: 88 }, (_, i) => pickedFile(`n${i}.md`, `docs/n${i}.md`)),
      ...Array.from({ length: 312 }, (_, i) => pickedFile(`img${i}.png`, `docs/img${i}.png`)),
    ];
    const plan = planAttachments(files, ALLOWED, 1, { folder: true });
    expect(plan.accepted).toHaveLength(88);
    expect(plan.paths).toHaveLength(88);
    expect(summarizeSkipped(plan.skipped)).toBe("312 files skipped — unsupported type");
  });

  it("buckets hidden, oversized, and lookalike paths separately", () => {
    const plan = planAttachments(
      [
        pickedFile("config", "repo/.git/config"),
        pickedFile("big.md", "repo/big.md", 2 * 1024 * 1024),
        pickedFile("weird.md", "repo／weird.md"), // fullwidth solidus survives the size/ext checks
        pickedFile("ok.md", "repo/ok.md"),
      ],
      ALLOWED,
      1,
      { folder: true },
    );
    expect(plan.accepted.map((file) => file.name)).toEqual(["ok.md"]);
    expect(plan.skipped).toEqual({ hidden: 1, "too large": 1, "invalid path": 1 });
  });

  it("keeps explicit file picks flat (no paths field)", () => {
    const plan = planAttachments(
      [pickedFile("a.md"), pickedFile("virus.exe")],
      ALLOWED,
      1,
      { folder: false },
    );
    expect(plan.accepted.map((file) => file.name)).toEqual(["a.md"]);
    expect(plan.paths).toBeNull();
    expect(plan.skipped).toEqual({ "unsupported type": 1 });
  });

  it("skips extension/size checks while config is unloaded, but still path-filters", () => {
    const plan = planAttachments(
      [pickedFile("anything.xyz", "docs/anything.xyz"), pickedFile("x", "docs/.hidden/x")],
      null,
      null,
      { folder: true },
    );
    expect(plan.accepted.map((file) => file.name)).toEqual(["anything.xyz"]);
    expect(plan.skipped).toEqual({ hidden: 1 });
  });
});

describe("summarizeSkipped", () => {
  it("is null when nothing was skipped", () => {
    expect(summarizeSkipped({})).toBeNull();
    expect(summarizeSkipped({ hidden: 0 })).toBeNull();
  });

  it("names the single bucket directly", () => {
    expect(summarizeSkipped({ hidden: 1 })).toBe("1 file skipped — hidden");
    expect(summarizeSkipped({ "unsupported type": 3 })).toBe(
      "3 files skipped — unsupported type",
    );
  });

  it("breaks down multiple buckets", () => {
    expect(summarizeSkipped({ "unsupported type": 312, hidden: 2 })).toBe(
      "314 files skipped — 312 unsupported type, 2 hidden",
    );
  });
});

describe("skipLabel", () => {
  it("maps client and server reasons onto display buckets", () => {
    expect(skipLabel("extension_not_allowed")).toBe("unsupported type");
    expect(skipLabel("file_too_large")).toBe("too large");
    expect(skipLabel("invalid_filename")).toBe("invalid path");
    expect(skipLabel("invalid_path")).toBe("invalid path");
    expect(skipLabel("path_too_deep")).toBe("folder too deep");
    expect(skipLabel("hidden")).toBe("hidden");
    expect(skipLabel("write_failed")).toBe("rejected");
  });
});
