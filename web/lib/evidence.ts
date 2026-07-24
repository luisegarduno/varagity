/**
 * The evidence panel's normalized data model.
 *
 * The panel renders from two wire shapes: the live `retrieval` SSE event
 * (`RetrievedChunk` + full metadata record) and a persisted assistant
 * turn's `message_sources` snapshots (the server's `_source_snapshot`
 * dict). Both normalize into one {@link Evidence} shape here so every
 * component downstream has a single contract, and a just-streamed turn
 * renders exactly like the same turn reloaded from history.
 */
import type {
  ChatMessage,
  DoneEvent,
  RetrievalEvent,
  RetrievalTrace,
} from "@/lib/api";

/** One evidence row: a retrieved chunk with its provenance, display-ready. */
export interface EvidenceChunk {
  /** Unique key within one answer's evidence (the chunk id). */
  key: string;
  /**
   * Parent document id — from the live event's `doc_id`, or parsed out of
   * `chunk_id` (which embeds `{doc_id}::{chunk_index}`) for persisted
   * snapshots, so all existing history resolves one without a migration.
   * Gates the page-preview affordance; `null` when unparseable.
   */
  docId: string | null;
  /** Final 1-based rank in the answer's evidence. */
  rank: number;
  /** Final score (cross-encoder relevance when reranked, else fused/arm). */
  score: number | null;
  /** Original chunk text. */
  content: string;
  /** The Contextual-Retrieval situating blurb (`null` when ingested off). */
  context: string | null;
  /** Absolute source path (provenance; also the `[SOURCE]` cite target). */
  source: string | null;
  /** Basename of the source file. */
  fileName: string | null;
  /** `pdf` / `txt` / `md` / … — the format badge. */
  fileType: string | null;
  /** Page number when the format has one (`null` otherwise). */
  page: number | null;
  /** `"text"`, `"ocr"` (image), or `"ocr_fallback"` (PDF) — the OCR badge signal. */
  extraction: string | null;
  /** Source file's birth time (ISO; best-effort — often unavailable). */
  fileCreatedAt: string | null;
  /** Source file's mtime (ISO) — the document's clock, not the ingest's. */
  fileModifiedAt: string | null;
  /** Why it ranked where it did (`null` when the retriever attached none). */
  trace: RetrievalTrace | null;
}

/**
 * A completed turn's token accounting, session-only by design: it rides
 * the `done` event and is never persisted, so history reloaded from the
 * server shows latency but no counts (see `lib/session-usage.ts`).
 */
export interface EvidenceUsage {
  /** Server-reported prompt tokens (`null` when unreported). */
  promptTokens: number | null;
  /** Server-reported completion tokens (`null` when unreported). */
  completionTokens: number | null;
  /**
   * Final decode throughput as the model server measured it. `null` on
   * servers that report no timings — only llama.cpp does, which is what
   * scopes the tok/s readout to llama.cpp-hosted models.
   */
  tokensPerSecond: number | null;
}

/** One answer's full evidence: the chunks plus the answer-level meta. */
export interface Evidence {
  /** Which answer this belongs to: a persisted `message_id`, or `"live"`. */
  key: string;
  /** The question that produced the answer (drives term highlighting). */
  query: string | null;
  /**
   * The standalone search query the chat engine rewrote the turn into
   * (spec_v3 §4.7) — the "Searched for: …" line. `null` whenever the
   * search used the user's words verbatim.
   */
  condensedQuery: string | null;
  /** The evidence rows, best first. */
  chunks: EvidenceChunk[];
  /** Retrieval method that produced them. */
  method: string | null;
  /** Chunks requested from the retriever (live event only). */
  topK: number | null;
  /** `RERANK_TOP_N` when the reranked method narrowed the list. */
  rerankedTo: number | null;
  /** Wall-clock per-stage timings (`retrieval` / `generation` / `total`). */
  latencyMs: Record<string, number> | null;
  /** Token counts + decode rate (this session's turns only; else `null`). */
  usage: EvidenceUsage | null;
}

/** The `"live"` evidence key of the in-flight streaming turn. */
export const LIVE_EVIDENCE_KEY = "live";

function asString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/**
 * Recover the parent `doc_id` from a `{doc_id}::{chunk_index}` chunk id.
 *
 * Persisted `message_sources` rows never stored `doc_id`, but every
 * `chunk_id` embeds it (`varagity/stores/records.py`), so history gains
 * preview eligibility retroactively. The 16-hex guard rejects anything
 * that isn't a content-hash prefix rather than guessing.
 */
export function docIdFromChunkId(chunkId: string): string | null {
  const [id] = chunkId.split("::");
  return /^[0-9a-f]{16}$/.test(id) ? id : null;
}

/** Coerce a JSONB timing dict into a numeric record (drops non-numbers). */
export function latencyRecord(
  value: { [key: string]: unknown } | null | undefined,
): Record<string, number> | null {
  if (!value) return null;
  const timings: Record<string, number> = {};
  for (const [stage, ms] of Object.entries(value)) {
    const parsed = asNumber(ms);
    if (parsed !== null) timings[stage] = parsed;
  }
  return Object.keys(timings).length > 0 ? timings : null;
}

/**
 * Normalize a `done` event's usage block, or `null` when the model server
 * reported neither counts nor timings (nothing worth a footer line).
 */
export function usageFromDone(usage: DoneEvent["usage"]): EvidenceUsage | null {
  const normalized: EvidenceUsage = {
    promptTokens: usage.prompt_tokens ?? null,
    completionTokens: usage.completion_tokens ?? null,
    tokensPerSecond: usage.tokens_per_second ?? null,
  };
  return Object.values(normalized).some((value) => value !== null)
    ? normalized
    : null;
}

/**
 * Display form of a decode rate: whole tokens/second, one decimal only
 * below 10 where rounding would hide most of the number.
 */
export function formatTokensPerSecond(rate: number): string {
  const figure = rate >= 10 ? Math.round(rate).toString() : rate.toFixed(1);
  return `${figure} tok/s`;
}

function asDate(value: string | null): Date | null {
  if (value === null) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

/**
 * The provenance line's file-clock segment: a short local date for the
 * source file's last modification, with the full timestamp (and the birth
 * time, when the filesystem recorded one) in the tooltip. `null` for
 * chunks ingested before the fields existed, or when the value doesn't
 * parse — the line simply omits the segment.
 */
export function fileClock(
  chunk: Pick<EvidenceChunk, "fileCreatedAt" | "fileModifiedAt">,
): { text: string; title: string } | null {
  const modified = asDate(chunk.fileModifiedAt);
  if (modified === null) return null;
  const created = asDate(chunk.fileCreatedAt);
  const title =
    `Source file modified ${modified.toLocaleString()}` +
    (created === null ? "" : ` · created ${created.toLocaleString()}`);
  return { text: `modified ${modified.toLocaleDateString()}`, title };
}

/**
 * The "Show metadata" disclosure's rows (one card's raw provenance
 * record as label/value pairs, absent fields omitted). Deliberately
 * limited to the normalized {@link EvidenceChunk} fields — the ones both
 * wire shapes carry (live `retrieval` event and persisted
 * `message_sources` snapshot) — so a just-streamed turn and its reload
 * render the identical list. Timestamps localize; an unparseable one
 * falls back to the stored string (it's a raw view — show what's there).
 */
export function metadataRows(
  chunk: EvidenceChunk,
): { label: string; value: string }[] {
  const stamp = (iso: string): string => asDate(iso)?.toLocaleString() ?? iso;
  const record: [string, string | null][] = [
    ["Chunk ID", chunk.key],
    ["Document ID", chunk.docId],
    ["Source path", chunk.source],
    ["File name", chunk.fileName],
    ["File type", chunk.fileType],
    ["Page", chunk.page === null ? null : String(chunk.page)],
    ["Extraction", chunk.extraction],
    ["File created", chunk.fileCreatedAt === null ? null : stamp(chunk.fileCreatedAt)],
    ["File modified", chunk.fileModifiedAt === null ? null : stamp(chunk.fileModifiedAt)],
  ];
  return record
    .filter((entry): entry is [string, string] => entry[1] !== null)
    .map(([label, value]) => ({ label, value }));
}

/** A loosely-typed persisted snapshot's serialized retrieval trace. */
function traceFromSnapshot(value: unknown): RetrievalTrace | null {
  if (value === null || typeof value !== "object") return null;
  const raw = value as { [key: string]: unknown };
  const fusedScore = asNumber(raw.fused_score);
  const fusedRank = asNumber(raw.fused_rank);
  const finalRank = asNumber(raw.final_rank);
  if (fusedScore === null || fusedRank === null || finalRank === null) {
    return null;
  }
  return {
    semantic_rank: asNumber(raw.semantic_rank),
    semantic_score: asNumber(raw.semantic_score),
    bm25_rank: asNumber(raw.bm25_rank),
    bm25_score: asNumber(raw.bm25_score),
    fused_score: fusedScore,
    fused_rank: fusedRank,
    rerank_score: asNumber(raw.rerank_score),
    rerank_delta: asNumber(raw.rerank_delta),
    final_rank: finalRank,
  };
}

/**
 * Normalize the live `retrieval` SSE event into {@link Evidence}.
 *
 * Chunks arrive best-first, so the array position is the final rank; the
 * provenance fields live in each chunk's full metadata record.
 */
export function evidenceFromRetrieval(
  event: RetrievalEvent,
  options: {
    key?: string;
    query?: string | null;
    latencyMs?: Record<string, number> | null;
    usage?: EvidenceUsage | null;
  } = {},
): Evidence {
  return {
    key: options.key ?? LIVE_EVIDENCE_KEY,
    query: options.query ?? null,
    condensedQuery: event.condensed_query,
    method: event.method,
    topK: event.top_k,
    rerankedTo: event.reranked_to,
    latencyMs: options.latencyMs ?? null,
    usage: options.usage ?? null,
    chunks: event.chunks.map((chunk, index) => ({
      key: chunk.chunk_id,
      docId: chunk.doc_id,
      rank: index + 1,
      score: chunk.score,
      content: chunk.content,
      context: chunk.context,
      source: asString(chunk.metadata.source),
      fileName: asString(chunk.metadata.file_name),
      fileType: asString(chunk.metadata.file_type),
      page: asNumber(chunk.metadata.page),
      extraction: asString(chunk.metadata.extraction),
      fileCreatedAt: asString(chunk.metadata.file_created_at),
      fileModifiedAt: asString(chunk.metadata.file_modified_at),
      trace: chunk.trace,
    })),
  };
}

/**
 * Normalize a persisted assistant message's snapshotted sources.
 *
 * Returns `null` for user turns and for assistant turns with no stored
 * evidence (nothing for the panel to show). `top_k`/`reranked_to` are not
 * persisted — the trace's `rerank_delta` still marks reranked answers.
 * Token usage is not persisted either: `usage` is the caller's
 * session-recall lookup (`lib/session-usage.ts`), `null` for turns
 * answered before this page load.
 */
export function evidenceFromMessage(
  message: ChatMessage,
  query: string | null,
  usage: EvidenceUsage | null = null,
): Evidence | null {
  if (message.role !== "assistant" || message.sources.length === 0) {
    return null;
  }
  return {
    key: message.message_id,
    query,
    condensedQuery: message.condensed_query ?? null,
    method: message.retrieval_method ?? null,
    topK: null,
    rerankedTo: null,
    latencyMs: latencyRecord(message.latency_ms),
    usage,
    chunks: message.sources.map((row) => {
      const snapshot = row.trace;
      return {
        key: row.chunk_id,
        docId: docIdFromChunkId(row.chunk_id),
        rank: row.rank,
        score: asNumber(snapshot.score),
        content: asString(snapshot.content) ?? "",
        context: asString(snapshot.context),
        source: asString(snapshot.source),
        fileName: asString(snapshot.file_name),
        fileType: asString(snapshot.file_type),
        page: asNumber(snapshot.page),
        extraction: asString(snapshot.extraction),
        fileCreatedAt: asString(snapshot.file_created_at),
        fileModifiedAt: asString(snapshot.file_modified_at),
        trace: traceFromSnapshot(snapshot.trace),
      };
    }),
  };
}

/**
 * Build `message_sources`-shaped rows from the live `retrieval` event —
 * the client-side mirror of the server's `_source_snapshot`, so a turn
 * folded into the transcript at `done` carries the same evidence a reload
 * would fetch.
 */
export function sourcesFromRetrieval(
  event: RetrievalEvent,
): ChatMessage["sources"] {
  return event.chunks.map((chunk, index) => ({
    rank: index + 1,
    chunk_id: chunk.chunk_id,
    trace: {
      score: chunk.score,
      content: chunk.content,
      context: chunk.context,
      source: chunk.metadata.source ?? null,
      file_name: chunk.metadata.file_name ?? null,
      file_type: chunk.metadata.file_type ?? null,
      page: chunk.metadata.page ?? null,
      extraction: chunk.metadata.extraction ?? null,
      file_created_at: chunk.metadata.file_created_at ?? null,
      file_modified_at: chunk.metadata.file_modified_at ?? null,
      trace: chunk.trace,
    },
  }));
}

/**
 * Build the locally-persisted assistant message for a completed turn.
 *
 * The fold-at-`done` twin of the server's persistence: authoritative
 * answer, method, per-stage latency, captured reasoning, and the evidence
 * snapshot — so the just-answered turn renders identically to a reload.
 */
export function assistantMessageFromTurn(
  done: DoneEvent,
  retrieval: RetrievalEvent | null,
  reasoning: string,
): ChatMessage {
  return {
    message_id: done.message_id,
    role: "assistant",
    content: done.answer,
    created_at: new Date().toISOString(),
    retrieval_method: retrieval?.method ?? null,
    latency_ms: done.usage.latency_ms,
    reasoning: reasoning || null,
    condensed_query: retrieval?.condensed_query ?? null,
    sources: retrieval ? sourcesFromRetrieval(retrieval) : [],
  };
}
