/**
 * The evidence panel's normalized data model.
 *
 * The panel renders from two wire shapes: the live `retrieval` SSE event
 * (`RetrievedChunk` + full metadata record) and a persisted assistant
 * turn's `message_sources` snapshots (the server's `_source_snapshot`
 * dict). Both normalize into one {@link Evidence} shape here so every
 * component downstream has a single contract, and a just-streamed turn
 * renders exactly like the same turn reloaded from history.
 *
 * A **graph** turn (spec_graphrag §4.3) rides the same two shapes: the
 * live event's `graph` payload, or the persisted `graph_evidence` column
 * plus the `kind: "graph_transcript"` source rows. Its cited transcript
 * days normalize into the same {@link EvidenceChunk} list the chunk
 * corpus fills, keyed by `doc_key` and sourced by the day label — so
 * citation matching, the panel's scroll-to-card, and the evidence
 * snapshot all work through machinery that already existed. A graph turn
 * that *degraded* to a chunk answer normalizes as exactly that: `corpus`
 * `"graph"`, `graph` null, ordinary chunks.
 */
import type {
  ChatMessage,
  DoneEvent,
  RetrievalEvent,
  RetrievalTrace,
  TargetCorpus,
} from "@/lib/api";
import { asTargetCorpus, DEFAULT_CORPUS } from "@/lib/corpus";

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

/** One entity a graph retrieval surfaced. */
export interface GraphEvidenceEntity {
  name: string;
  /** The engine's category — what the chips are toned by (LightRAG has no
   * community tier, so `entity_type` is the grouping that exists). */
  type: string | null;
  summary: string | null;
}

/** One relation a graph retrieval surfaced — the *fact* an answer grounds on. */
export interface GraphEvidenceRelation {
  source: string;
  target: string;
  label: string | null;
  description: string | null;
}

/**
 * One cited transcript day. Day-grain, not message-grain: that is what
 * the engine can honestly attribute (ADR-017's priced regret), and the
 * panel says so rather than inventing per-message rows.
 */
export interface GraphEvidenceTranscript {
  /** The transcript document key — the join back to the graph corpus. */
  docKey: string;
  threadName: string;
  /** `YYYY-MM-DD`, or `first..last` for a multi-day document. */
  span: string;
  excerpt: string;
  /** Messages the graph's manifest accounts for (`0` when it cannot say). */
  messageCount: number;
  /**
   * A second label the same document was cited under, kept when duplicate
   * hits collapse ({@link dedupeTranscripts}): a doc-grain hit carries the
   * rendered thread name while a chunk-grain hit of the same document only
   * knows the thread guid, and the answer may cite either.
   */
  altLabel?: string;
}

/** The graph half of one answer's evidence (`null` for chunk-RAG turns). */
export interface GraphEvidence {
  /** The engine query mode that retrieved it (`null` in older snapshots). */
  mode: string | null;
  entities: GraphEvidenceEntity[];
  relations: GraphEvidenceRelation[];
  transcripts: GraphEvidenceTranscript[];
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
  /**
   * The evidence rows, best first — retrieved chunks for a chunk-RAG
   * turn, the cited transcript days for a graph one.
   */
  chunks: EvidenceChunk[];
  /**
   * Which corpus the turn asked for. Pre-stage-2 history carries no
   * value at all and reads as `"rag"`, which is what it was.
   */
  corpus: TargetCorpus;
  /**
   * The graph evidence, present only on a graph turn the engine actually
   * answered. `null` on a chunk turn *and* on a graph turn that degraded
   * — where `corpus` still says `"graph"` and the chunks say what really
   * ran (ADR-017's degrade semantics, visible rather than papered over).
   */
  graph: GraphEvidence | null;
  /**
   * Retrieval method that produced them, in the corpus's own vocabulary:
   * the retriever's registry name for a chunk turn, the engine query mode
   * for a graph one.
   */
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

/** The `message_sources.trace` discriminator a graph turn's rows carry. */
export const GRAPH_TRANSCRIPT_KIND = "graph_transcript";

/**
 * One cited transcript day's citation label.
 *
 * Must stay identical to `varagity/graph/answer.py::transcript_label`,
 * which is what the model is shown and therefore what it cites: the
 * server owns the format, this is the client's half of the same contract.
 */
export function transcriptLabel(
  transcript: Pick<GraphEvidenceTranscript, "threadName" | "span">,
): string {
  return `${transcript.threadName} (${transcript.span})`;
}

/**
 * Collapse duplicate transcript citations onto one entry per document.
 *
 * `mix` mode reaches the same transcript through more than one arm
 * (vector text-units and graph references), and the server keeps every
 * hit — so the same `docKey` can arrive twice, once labelled with the
 * rendered thread name (doc-grain) and once with the raw thread guid
 * (chunk-grain). One entry per document, best rank first; a
 * differently-labelled duplicate survives as `altLabel` so a citation
 * under either name still resolves. Persisted snapshots carry the
 * duplicates forever, which is why this lives in normalization rather
 * than upstream.
 */
function dedupeTranscripts(
  transcripts: GraphEvidenceTranscript[],
): GraphEvidenceTranscript[] {
  const byKey = new Map<string, GraphEvidenceTranscript>();
  for (const transcript of transcripts) {
    const seen = byKey.get(transcript.docKey);
    if (seen === undefined) {
      byKey.set(transcript.docKey, { ...transcript });
    } else if (
      seen.altLabel === undefined &&
      transcriptLabel(transcript) !== transcriptLabel(seen)
    ) {
      seen.altLabel = transcriptLabel(transcript);
    }
  }
  return [...byKey.values()];
}

/**
 * Render one cited transcript day as an evidence row.
 *
 * Deliberately chunk-shaped: `source`/`fileName` carry the day label the
 * answer cites, so `matchSource` resolves a graph citation with no
 * special case, and `docId` stays `null` so the page-preview affordance
 * (which needs an ingested document) never offers itself. A deduped
 * entry's `altLabel` rides in `fileName` — the other name the answer may
 * have cited the same document under.
 */
function chunkFromTranscript(
  transcript: GraphEvidenceTranscript,
  index: number,
): EvidenceChunk {
  const label = transcriptLabel(transcript);
  return {
    key: transcript.docKey,
    docId: null,
    rank: index + 1,
    score: null,
    content: transcript.excerpt,
    context: null,
    source: label,
    fileName: transcript.altLabel ?? label,
    fileType: null,
    page: null,
    extraction: null,
    fileCreatedAt: null,
    fileModifiedAt: null,
    trace: null,
  };
}

/** Render one live-event chunk as an evidence row (position = final rank). */
function chunkFromRetrieved(
  chunk: RetrievalEvent["chunks"][number],
  index: number,
): EvidenceChunk {
  return {
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
  };
}

function graphEntities(value: unknown): GraphEvidenceEntity[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((raw) => {
    if (raw === null || typeof raw !== "object") return [];
    const row = raw as { [key: string]: unknown };
    const name = asString(row.name);
    if (name === null) return [];
    return [{ name, type: asString(row.type), summary: asString(row.summary) }];
  });
}

function graphRelations(value: unknown): GraphEvidenceRelation[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((raw) => {
    if (raw === null || typeof raw !== "object") return [];
    const row = raw as { [key: string]: unknown };
    const source = asString(row.source);
    const target = asString(row.target);
    if (source === null || target === null) return [];
    return [
      {
        source,
        target,
        label: asString(row.label),
        description: asString(row.description),
      },
    ];
  });
}

/** Normalize the live event's graph payload (`null` for a chunk turn). */
function graphFromRetrieval(event: RetrievalEvent): GraphEvidence | null {
  if (!event.graph) return null;
  return {
    mode: event.graph.mode,
    entities: event.graph.entities.map((entity) => ({
      name: entity.name,
      type: entity.type ?? null,
      summary: entity.summary ?? null,
    })),
    relations: event.graph.relations.map((relation) => ({
      source: relation.source,
      target: relation.target,
      label: relation.label ?? null,
      description: relation.description ?? null,
    })),
    transcripts: dedupeTranscripts(
      event.graph.transcripts.map((transcript) => ({
        docKey: transcript.doc_key,
        threadName: transcript.thread_name,
        span: transcript.span,
        excerpt: transcript.excerpt,
        messageCount: transcript.message_count ?? 0,
      })),
    ),
  };
}

/**
 * Normalize a persisted graph turn: the `graph_evidence` JSONB column
 * plus the `graph_transcript` source rows.
 *
 * Returns `null` when neither is present — a chunk turn, or a graph turn
 * that degraded (whose record deliberately looks like the chunk answer it
 * really produced).
 */
function graphFromMessage(message: ChatMessage): GraphEvidence | null {
  const transcripts = dedupeTranscripts(
    message.sources.flatMap((row) => {
      const snapshot = row.trace;
      if (snapshot?.kind !== GRAPH_TRANSCRIPT_KIND) return [];
      const guids = snapshot.message_guids;
      return [
        {
          docKey: row.chunk_id,
          threadName: asString(snapshot.thread_name) ?? row.chunk_id,
          span: asString(snapshot.span) ?? "",
          excerpt: asString(snapshot.excerpt) ?? "",
          // The server snapshots the guids themselves; the client's
          // fold-at-`done` mirror only knows the count the wire carried.
          messageCount:
            asNumber(snapshot.message_count) ??
            (Array.isArray(guids) ? guids.length : 0),
        },
      ];
    }),
  );
  const snapshot = message.graph_evidence ?? null;
  if (snapshot === null && transcripts.length === 0) return null;
  return {
    mode: snapshot === null ? null : asString(snapshot.mode),
    entities: graphEntities(snapshot?.entities),
    relations: graphRelations(snapshot?.relations),
    transcripts,
  };
}

/**
 * Normalize the live `retrieval` SSE event into {@link Evidence}.
 *
 * Chunks arrive best-first, so the array position is the final rank; the
 * provenance fields live in each chunk's full metadata record. A graph
 * turn's `chunks` come from its cited transcript days instead — the event
 * leaves the chunk list empty and fills `graph`.
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
  const graph = graphFromRetrieval(event);
  return {
    key: options.key ?? LIVE_EVIDENCE_KEY,
    query: options.query ?? null,
    condensedQuery: event.condensed_query,
    corpus: asTargetCorpus(event.corpus) ?? DEFAULT_CORPUS,
    graph,
    method: event.method,
    topK: event.top_k,
    rerankedTo: event.reranked_to,
    latencyMs: options.latencyMs ?? null,
    usage: options.usage ?? null,
    chunks: graph
      ? graph.transcripts.map(chunkFromTranscript)
      : event.chunks.map(chunkFromRetrieved),
  };
}

/**
 * Normalize a persisted assistant message's snapshotted sources.
 *
 * Returns `null` for user turns and for assistant turns with no stored
 * evidence (nothing for the panel to show) — a graph turn counting as
 * evidence-bearing whenever it kept *either* transcript rows or the
 * `graph_evidence` snapshot, since entities and relations are worth a
 * panel even when the retrieval cited no day. `top_k`/`reranked_to` are
 * not persisted — the trace's `rerank_delta` still marks reranked
 * answers. Token usage is not persisted either: `usage` is the caller's
 * session-recall lookup (`lib/session-usage.ts`), `null` for turns
 * answered before this page load.
 */
export function evidenceFromMessage(
  message: ChatMessage,
  query: string | null,
  usage: EvidenceUsage | null = null,
): Evidence | null {
  if (message.role !== "assistant") return null;
  const graph = graphFromMessage(message);
  if (message.sources.length === 0 && graph === null) return null;
  return {
    key: message.message_id,
    query,
    condensedQuery: message.condensed_query ?? null,
    corpus: asTargetCorpus(message.target_corpus) ?? DEFAULT_CORPUS,
    graph,
    method: message.retrieval_method ?? null,
    topK: null,
    rerankedTo: null,
    latencyMs: latencyRecord(message.latency_ms),
    usage,
    chunks: graph
      ? graph.transcripts.map(chunkFromTranscript)
      : message.sources.map((row) => {
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
 *
 * A graph turn writes the other row shape (`kind: "graph_transcript"`,
 * `chunk_id` = the transcript day's `doc_key`), matching
 * `_graph_source_snapshot` — with the wire's `message_count` standing in
 * for the guid list the server has and the browser was never sent.
 */
export function sourcesFromRetrieval(
  event: RetrievalEvent,
): ChatMessage["sources"] {
  if (event.graph) {
    return event.graph.transcripts.map((transcript, index) => ({
      rank: index + 1,
      chunk_id: transcript.doc_key,
      trace: {
        kind: GRAPH_TRANSCRIPT_KIND,
        thread_name: transcript.thread_name,
        span: transcript.span,
        excerpt: transcript.excerpt,
        message_count: transcript.message_count,
      },
    }));
  }
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
 *
 * `corpus` is what the *request* asked for, and is stamped even when the
 * turn never reached its retrieval event (an error before the stream
 * opened): the server records the request the same way, and the
 * composer's source selector derives its value from this field.
 */
export function assistantMessageFromTurn(
  done: DoneEvent,
  retrieval: RetrievalEvent | null,
  reasoning: string,
  corpus: TargetCorpus = DEFAULT_CORPUS,
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
    target_corpus: asTargetCorpus(retrieval?.corpus) ?? corpus,
    graph_evidence: retrieval?.graph
      ? {
          mode: retrieval.graph.mode,
          entities: retrieval.graph.entities,
          relations: retrieval.graph.relations,
        }
      : null,
    sources: retrieval ? sourcesFromRetrieval(retrieval) : [],
  };
}
