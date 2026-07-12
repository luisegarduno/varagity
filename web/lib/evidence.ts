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
  /** `"text"` or `"ocr_fallback"` — the OCR badge signal. */
  extraction: string | null;
  /** Why it ranked where it did (`null` when the retriever attached none). */
  trace: RetrievalTrace | null;
}

/** One answer's full evidence: the chunks plus the answer-level meta. */
export interface Evidence {
  /** Which answer this belongs to: a persisted `message_id`, or `"live"`. */
  key: string;
  /** The question that produced the answer (drives term highlighting). */
  query: string | null;
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
}

/** The `"live"` evidence key of the in-flight streaming turn. */
export const LIVE_EVIDENCE_KEY = "live";

function asString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
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
  } = {},
): Evidence {
  return {
    key: options.key ?? LIVE_EVIDENCE_KEY,
    query: options.query ?? null,
    method: event.method,
    topK: event.top_k,
    rerankedTo: event.reranked_to,
    latencyMs: options.latencyMs ?? null,
    chunks: event.chunks.map((chunk, index) => ({
      key: chunk.chunk_id,
      rank: index + 1,
      score: chunk.score,
      content: chunk.content,
      context: chunk.context,
      source: asString(chunk.metadata.source),
      fileName: asString(chunk.metadata.file_name),
      fileType: asString(chunk.metadata.file_type),
      page: asNumber(chunk.metadata.page),
      extraction: asString(chunk.metadata.extraction),
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
 */
export function evidenceFromMessage(
  message: ChatMessage,
  query: string | null,
): Evidence | null {
  if (message.role !== "assistant" || message.sources.length === 0) {
    return null;
  }
  return {
    key: message.message_id,
    query,
    method: message.retrieval_method ?? null,
    topK: null,
    rerankedTo: null,
    latencyMs: latencyRecord(message.latency_ms),
    chunks: message.sources.map((row) => {
      const snapshot = row.trace;
      return {
        key: row.chunk_id,
        rank: row.rank,
        score: asNumber(snapshot.score),
        content: asString(snapshot.content) ?? "",
        context: asString(snapshot.context),
        source: asString(snapshot.source),
        fileName: asString(snapshot.file_name),
        fileType: asString(snapshot.file_type),
        page: asNumber(snapshot.page),
        extraction: asString(snapshot.extraction),
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
    sources: retrieval ? sourcesFromRetrieval(retrieval) : [],
  };
}
