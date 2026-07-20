# ADR-011: The chat engine registry and Condense + Context

**Status:** Accepted (2026-07-19)

## Context

v2's chat was stateless at every layer, verified before v3 started: the
route fed the flows exactly one string, the answer prompt had one
`{query}` slot, and history — though persisted since
[ADR-005 §4](ADR-005-web-stack-and-api.md) — was never read back into a
turn. A follow-up like *"how long is it?"* retrieved against the pronoun,
not the referent. v3's centerpiece (spec_v3 §4) is making the chat
remember, without rewriting what the user actually asked.

Three shapes were on the table: widen the `Retriever` protocol with a
history parameter, hang a boolean condense flag inside the chat route, or
open a third registry beside parsers/chunkers/retrievers. House
constraints applied: pluggable families are registries (spec §5.1),
defaults are benchmark-decided ([ADR-004](ADR-004-ocr-engine-choice.md),
[ADR-008](ADR-008-chunking-default.md)), and every pipeline stage is a
tracked Prefect task.

## Decision

- **A chat-engine registry** (`varagity/chat/`), mirroring
  `varagity/retrieval/` exactly (the PEP 695 `register` decorator,
  self-registration on package import, `get_chat_engine` raising
  `KeyError` with the available names). A `ChatEngine` decides **what
  string the retriever searches with**, given the turn and its history.
  v3 ships exactly two: `simple` (the v2 behavior verbatim — the identity
  split, no LLM call) and `condense_context` (LlamaIndex's
  Condense-Plus-Context pattern adapted to the registry convention): one
  non-streaming LLM call rewrites a follow-up into a standalone search
  query against the last `CONDENSE_HISTORY_TURNS` turns.
- **The two-string split is the invariant** (`PreparedQuery`):
  `search_query` drives retrieval — both the query embedding and BM25 —
  while `original_query`, always the user's words, is what the answer
  prompt gets. And if we searched with something other than what you
  typed, **you get to see it**: the SSE `retrieval` event carries
  `condensed_query`, the evidence panel renders it as "Searched for: …",
  and migrations 003/004 persist `condensed_query` + `chat_engine` per
  assistant message so historical conversations explain themselves — the
  same snapshot semantics as `message_sources.trace`.
- **The condense stage is always in the graph**: `condense_query` is a
  tracked task (`NO_CACHE`, and **no Prefect retries** — the query-path
  convention) even under `simple`, which returns the identity in ~3 ms.
  One extra task run per query is the cost of "every stage is tracked",
  and it proves the registry isn't a special case built for one engine.
- **Failure is a fallback, not an error** (spec_v3 §4.6): a transient LLM
  failure (after the client's own `tenacity` retries), an empty result,
  or an over-length one (`CONDENSE_MAX_CHARS`) all degrade to searching
  with the raw query at `WARNING` — a degraded search query still answers
  a lot of questions; a 500 answers none. `CONDENSE_ENABLED=false` is the
  kill switch, checked *inside* `prepare` — deliberately orthogonal to
  `CHAT_ENGINE`, exactly the `RERANK_ENABLED` shape
  ([ADR-006](ADR-006-reranking-wired.md)). A degraded turn still persists
  its engine name with `condensed_query` NULL — the honest record of what
  happened.
- **The first turn never condenses**: empty history is the identity path,
  no LLM call. History loads bounded **in SQL**
  (`ConversationStore.recent_turns`: role + content only,
  `ORDER BY created_at DESC … LIMIT n`, no sources join) — never through
  `get_conversation`, which hydrates every message's JSONB trace
  snapshots on what is now the pre-TTFT hot path.
- **The shipped default stays `simple`** — decided by the numbers below,
  an owner-accepted measured "no" (2026-07-19).

## Rationale

- **Why not a `Retriever`**: condensing needs chat history, which doesn't
  fit `Retriever.retrieve(query, k, …)` — widening the signature would
  force a parameter on three retrievers that will never use it. Reranking
  could hide behind that protocol precisely because it needs nothing
  beyond `query` and `k`; a chat engine is a different axis (what to
  search for, not how to search), so it gets its own protocol.
- **Why a registry, not a flag**: the next engine (multi-hop, agentic
  retrieval — the deferred roadmap) becomes one file plus an import line,
  and `simple` stays a *measured baseline* rather than dead code — the
  `eval chat` harness enumerates the registry, so every engine added is
  automatically in the comparison.
- **Why the original query reaches the answer prompt**: the rewrite is
  retrieval metadata, not the question. The user's own words and emphasis
  must never be laundered through a condenser — and a bad rewrite may
  cost recall, but it can never misstate what was asked.
- **Why `condensed_query` extends `RetrievalEvent`** rather than minting
  an SSE event: the condensed query *is* retrieval metadata (it belongs
  beside `method` and `top_k`), the `retrieval` frame already lands
  before any prose, and one optional field costs less than a new event
  name every client must learn. Reusing `reasoning` was rejected — that
  event means "`<think>` content from the *answer* call", an unrelated
  trace.
- **The eval decided the default**
  (`data/eval/results/20260719T081214Z-chat.json` — 10 hand-built
  multi-turn fixtures, 21 turns, 11 follow-ups; scripted assistant
  replies so both engines see byte-identical history; each turn's
  `search_query` scored fact-anchored under `hybrid` and `reranked` at
  k ∈ {1, 3, 5}):
    - Follow-ups under `reranked`, condense vs simple: recall@1 **0.727
      vs 0.545**, @3 0.909 vs 0.818, @5 1.000 vs 0.909. The pronoun
      slice is decisive — **1.000 vs 0.600 at @1**, every condensed
      pronoun turn ranking its chunk first, while *"How long is it?"* is
      not found at all under `simple`. Bare `hybrid` widens the gap
      (@1 0.455 vs 0.182) — reranking masks part of what condensing wins.
    - Topic shifts: identical 1.000s — the condenser never dragged the
      old topic along (the spec_v3 §13.4 failure mode; the prompt's
      "do not carry the old topic" line is load-bearing and the fixtures
      test it explicitly). Elliptical refinements: a wash — 2 of 3 fell
      back safely on an empty condense, and the one rewrite ranked 3,
      equal to the raw query.
    - **Condense latency: mean 8.594 s, max 10.913 s per call** on the
      single 2080 Ti serving a *reasoning* model — thinking tokens
      dominate the budget. Against the ~9.3 s generate stage that nearly
      doubles time-to-first-token on follow-ups. The quality win is real;
      the latency is material; `CHAT_ENGINE` ships `simple`. The recorded
      revisit path is `CONDENSE_MODEL_TYPE` — a config seam (the model
      registry's `default`/`reasoning`/`tool` aliases all resolve to the
      one served LLM today), so pointing the condenser at a small
      non-thinking model when one lands is a setting, not a code change.
- **The eval caught two engine bugs** — the reason a discriminating
  fixture set was the eval's actual deliverable: (1)
  `CONDENSE_MAX_TOKENS=128` starved the reasoning model — every call hit
  `finish_reason: "length"` with the entire budget inside `<think>` and
  `content: ""`, so 0/11 follow-ups condensed and every turn silently
  degraded through the §4.6 fallback (working as designed, masking the
  bug); the default is now 512. (2) The model echoes the prompt's
  trailing `STANDALONE QUERY:` label into its answer; the engine strips
  it case-insensitively, with a drift test pinning template ↔ strip
  agreement.

## Consequences

- The chat remembers: a pronoun follow-up retrieves what it refers to
  (under `condense_context`), and the answer still addresses the user's
  literal question. The CLI gains the same via in-memory history —
  session-scoped, cleared by `:quit`, never persisted.
- **The condenser's output must pass through `clean_response()` plus the
  label strip before it reaches the embedder.** `LLMClient.generate()`
  returns reasoning tags verbatim — only the *streaming* path normalizes
  `reasoning_content` — so an unstripped `<think>` block would ride
  straight into e5 as the search query: the single easiest way to
  silently destroy retrieval quality in this feature. A named unit test
  guards it.
- Migrations `003_condensed_query.sql` / `004_message_engine.sql` add two
  nullable `messages` columns; NULL means "not condensed" (first turn,
  `simple`, kill switch, or fallback). `MessageOut` carries both, so a
  reloaded conversation renders the same "Searched for: …" line the live
  turn showed.
- `CHAT_ENGINE` and the `CONDENSE_*` knobs joined the runtime-settings
  overridable set (they are **not** reingest-affecting — query-time
  behavior only), and `ChatOverrides.chat_engine` allows per-request
  selection (`422 unknown_chat_engine` for a name the registry lacks).
- Per-request state stays off the registry singletons: the route observes
  the `PreparedQuery` through a delegating per-request wrapper to fill
  the SSE field — engines remain shared and stateless.
- The wire order is unchanged (`retrieval → deltas → done`); the payload
  is extended, not reshaped — one regeneration of `openapi.json` and the
  generated frontend types, in the same commit as the behavior.
- Rejected for v3, seams left open: rolling summarization of older turns
  (the bound is `CONDENSE_HISTORY_TURNS`; summarize-instead-of-drop is
  post-v3), CLI conversation persistence, multi-hop retrieval (the
  registry is exactly where it plugs in).
