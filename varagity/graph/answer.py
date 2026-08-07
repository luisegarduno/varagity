"""Grounded answer synthesis over graph evidence (spec_graphrag §7; ADR-017).

ADR-017 chose LightRAG **retrieval-only**: the engine finds the evidence and
*this repo* writes the answer, so a graph answer obeys the same grounding
discipline as a chunk-RAG one. This module is that seam, shared by every
engine adapter rather than reimplemented per engine — it started as the
Graphiti adapter's private synthesis (the fact-shaped candidate had no answer
pipeline of its own) and was generalized when it became the shipped design.

Two renderers, one diet: :func:`synthesis_context` grounds the harness's
non-streaming :func:`synthesize` (and the adapter's ``+synthesis`` modes),
while :func:`graph_answer_context` renders the *same* facts and passages in
the chunk-RAG ``[SOURCE]`` block shape the streamed API path answers from.
Keeping the two payloads identical is what lets ``eval graph`` keep guarding
the shipped query path.

Two rules run through it:

* **The context is the whole retrieval payload**, not just relation facts:
  entities and relations answer "who/what", but the *transcript excerpts* are
  where the archive actually says things (plan decision #6). An engine that
  retrieves both and is scored on facts alone under-measures itself.
* **Nothing here raises.** A failed synthesis returns ``""`` — a scored miss
  in the harness and a degrade-to-honest turn in the app, never a dead run.
  An empty context short-circuits to :data:`NO_EVIDENCE_ANSWER` without
  spending an LLM call at all.

Sizing follows the same clamp discipline as
:func:`varagity.graph.engines.lightrag.fit_max_tokens`: llama.cpp with
context shift disabled hard-500s at the window instead of stopping
gracefully, and a retrieval payload is bounded by the *engine's* token budget
(LightRAG's ``max_total_tokens`` defaults far above a 16k window), so the
cap has to be applied here — see :func:`synthesis_max_chars`.
"""

import logging
from collections.abc import Callable, Sequence

from varagity.generation.answer import format_source_block
from varagity.graph.records import GraphEvidence, GraphRetrieval, TranscriptExcerpt
from varagity.models.llm import LLMClient, clean_response

logger = logging.getLogger(__name__)

SYNTHESIS_PROMPT = """\
You are answering a question about a personal message archive, using only the
evidence retrieved from its knowledge graph.

Rules:
- Use ONLY the evidence below. Do not invent people, events, dates, or opinions.
- If the evidence does not answer the question, say so plainly.
- Answer in a few sentences, naming the people involved.

{context}

QUESTION: {question}

ANSWER:"""

# The honest answer when retrieval surfaced nothing at all. Returned without
# calling the model: there is nothing to ground on, and the harness scores
# the empty retrieval rather than a hallucination.
NO_EVIDENCE_ANSWER = "The graph returned no facts for this question."

# Cap for the synthesis call — an answer, not a document. The cap covers the
# *whole* generation, and a reasoning model spends most of it inside its
# reasoning stage before the answer starts: at 1024 the served model burned
# the entire budget thinking on ~half of gate-run calls, and llama.cpp
# routes an unfinished think phase into `reasoning_content`, so
# `generate()`'s content came back "" (the condense 128→512 and HyDE
# 512→1024 precedents, one size larger because the evidence context here is
# thousands of tokens).
SYNTHESIS_MAX_TOKENS = 2048

# Room reserved for everything in the prompt that is not the context: the
# template, the question, the chat scaffolding, and the drift between the
# cl100k approximation and the served model's own tokenizer.
_PROMPT_HEADROOM_TOKENS = 1024

# Deliberately pessimistic: transcript lines are timestamps, handles, and
# punctuation, which tokenize worse than the ~4 chars/token of prose.
_CHARS_PER_TOKEN = 3

# A context this small cannot ground an answer, but a floor keeps a
# misconfigured window from silently producing an empty prompt.
_MIN_CONTEXT_CHARS = 1000

_FACTS_HEADER = "FACTS:"
_EXCERPTS_HEADER = "TRANSCRIPT EXCERPTS:"
# The facts header of the *streamed* path, whose context is read under
# varagity.generation.answer.ANSWER_PROMPT ("cite the [SOURCE] of any facts
# you use"). Deliberately not a [SOURCE] block: relations are derived from
# the transcripts below them, so a citable label here would produce chips
# the evidence panel has no card for.
_GRAPH_FACTS_HEADER = "GRAPH FACTS (relations extracted from the transcripts below):"
_SECTION_SEPARATOR = "\n\n"
_ELLIPSIS = " […]"


def answer_context_max_chars(context_tokens: int, generation_tokens: int) -> int:
    """Size a grounding context so prompt + answer fits the context window.

    Args:
        context_tokens: The served model's context window
            (``LLM_CONTEXT_TOKENS``).
        generation_tokens: The generation cap the answer call will request.

    Returns:
        The character budget, never below :data:`_MIN_CONTEXT_CHARS`.
    """
    budget = context_tokens - generation_tokens - _PROMPT_HEADROOM_TOKENS
    return max(_MIN_CONTEXT_CHARS, budget * _CHARS_PER_TOKEN)


def synthesis_max_chars(context_tokens: int) -> int:
    """Size the synthesis context so prompt + answer fits the context window.

    Args:
        context_tokens: The served model's context window
            (``LLM_CONTEXT_TOKENS``).

    Returns:
        The character budget for :func:`synthesis_context`, never below
        :data:`_MIN_CONTEXT_CHARS`.
    """
    return answer_context_max_chars(context_tokens, SYNTHESIS_MAX_TOKENS)


def transcript_label(excerpt: TranscriptExcerpt) -> str:
    """Render the citation label of one transcript passage.

    The one owner of the label, because it is matched three ways: the model
    cites it out of the prompt, the evidence panel keys its cards by it, and
    the persisted ``message_sources`` snapshot stores it. A day-grain label
    is also the ADR-017 honesty line — the engines cannot attribute a single
    message, so nothing here pretends to.

    Args:
        excerpt: The retrieved passage.

    Returns:
        ``"{thread_name} ({span})"``.
    """
    return f"{excerpt.thread_name} ({excerpt.span})"


def facts_block(evidence: GraphEvidence) -> str:
    """Render retrieved relations and communities as prompt facts.

    Args:
        evidence: The normalized evidence from a graph retrieval.

    Returns:
        One ``- fact`` line per relation (community summaries appended after
        them), or ``""`` when the retrieval surfaced no describable edge —
        which is what makes the synthesis call skippable.
    """
    lines = [
        f"- {relation.description or relation.label or ''}".rstrip()
        for relation in evidence.relations
        if relation.description or relation.label
    ]
    lines.extend(
        f"- community {community.title or community.id}: {community.summary}"
        for community in evidence.communities
        if community.summary
    )
    return "\n".join(lines)


def excerpts_block(excerpts: Sequence[TranscriptExcerpt], *, max_chars: int) -> str:
    """Render retrieved transcript passages as labelled prompt blocks.

    Whole excerpts are packed until the next one would not fit; a single
    excerpt larger than the whole budget is truncated rather than dropped
    (one oversized passage must not cost the answer its only grounding). The
    label is ``[{thread_name} ({span})]`` — the same
    :func:`transcript_label` the shipped query path cites with, so the model
    learns one source format.

    Args:
        excerpts: Retrieved passages, most relevant first (the engine's
            order is preserved: packing drops from the tail).
        max_chars: Character budget for the whole block.

    Returns:
        The blocks, blank-line separated, never longer than ``max_chars``.
    """
    return _pack(
        excerpts,
        max_chars=max_chars,
        render=lambda excerpt, body: f"[{transcript_label(excerpt)}]\n{body}",
    )


def transcript_blocks(excerpts: Sequence[TranscriptExcerpt], *, max_chars: int) -> str:
    """Render retrieved transcript passages as ``[SOURCE]`` provenance blocks.

    :func:`excerpts_block`'s twin for the *streamed* answer path, which is
    read under :data:`varagity.generation.answer.ANSWER_PROMPT` rather than
    :data:`SYNTHESIS_PROMPT`: same packing, same labels, but the chunk-RAG
    block shape, so a graph turn's citations, chips, and evidence-card
    matching all work through the machinery chunk turns already use.

    Args:
        excerpts: Retrieved passages, most relevant first.
        max_chars: Character budget for the whole block.

    Returns:
        The blocks, blank-line separated, never longer than ``max_chars``.
    """
    return _pack(
        excerpts,
        max_chars=max_chars,
        render=lambda excerpt, body: format_source_block(
            source=transcript_label(excerpt), context="", content=body
        ),
    )


def _pack(
    excerpts: Sequence[TranscriptExcerpt],
    *,
    max_chars: int,
    render: Callable[[TranscriptExcerpt, str], str],
) -> str:
    """Fit as many rendered excerpts as a character budget holds.

    Args:
        excerpts: Retrieved passages, most relevant first (packing drops
            from the tail, keeping the engine's relevance order).
        max_chars: Character budget for the whole block.
        render: Renders one excerpt's block around a body; called with an
            empty body to measure the block's own overhead.

    Returns:
        The rendered blocks, blank-line separated, never longer than
        ``max_chars``.
    """
    blocks: list[str] = []
    used = 0
    for excerpt in excerpts:
        cost = used + (len(_SECTION_SEPARATOR) if blocks else 0)
        room = max_chars - cost - len(render(excerpt, ""))
        if room <= 0:
            break
        body = _truncate(excerpt.text.strip(), room)
        if not body:
            break
        block = render(excerpt, body)
        used = cost + len(block)
        blocks.append(block)
    return _SECTION_SEPARATOR.join(blocks)


def synthesis_context(
    evidence: GraphEvidence,
    excerpts: Sequence[TranscriptExcerpt] = (),
    *,
    max_chars: int,
) -> str:
    """Render one retrieval into the grounding context of a synthesis prompt.

    Facts come first (they are small and they name the people), then as many
    transcript excerpts as the remaining budget holds. Empty sections are
    omitted entirely, so a retrieval that surfaced nothing renders to ``""``
    and :func:`synthesize` can skip the model call.

    Args:
        evidence: Entities, relations, and communities the engine retrieved.
        excerpts: Transcript passages the engine retrieved (empty for
            fact-shaped engines, which have no document plane).
        max_chars: Character budget for the whole context — size it with
            :func:`synthesis_max_chars`.

    Returns:
        The context, never longer than ``max_chars``.
    """
    sections: list[str] = []
    used = 0
    facts = facts_block(evidence)
    if facts:
        # A budget too small even for the header leaves no section at all —
        # appending "" here would put a stray separator in front of the
        # excerpts.
        section = _truncate(f"{_FACTS_HEADER}\n{facts}", max_chars)
        if section:
            sections.append(section)
            used = len(section)
    room = max_chars - used - len(_EXCERPTS_HEADER) - 1
    if sections:
        room -= len(_SECTION_SEPARATOR)
    rendered = excerpts_block(excerpts, max_chars=max(0, room))
    if rendered:
        sections.append(f"{_EXCERPTS_HEADER}\n{rendered}")
    return _SECTION_SEPARATOR.join(sections)


def graph_answer_context(retrieval: GraphRetrieval, *, max_chars: int) -> str:
    """Render one graph retrieval into the chunk-RAG grounding context.

    :func:`synthesis_context`'s twin for the shipped streaming path
    (stage-2 decision #5): the same diet — relation facts first, then as
    many transcript passages as the budget holds — rendered in the shape
    :data:`varagity.generation.answer.ANSWER_PROMPT` expects, so a graph
    turn inherits the repo's grounding prompt, ``<think>`` splitting,
    citation contract, abort handling, and token accounting whole.

    The diets are deliberately identical: the ``eval graph`` harness scores
    :func:`synthesis_context`, and it can only keep guarding this path while
    both ground on the same evidence.

    Args:
        retrieval: What the engine found (entities and relations plus the
            transcript passages).
        max_chars: Character budget for the whole context — size it with
            :func:`answer_context_max_chars`.

    Returns:
        The context block, never longer than ``max_chars``, and ``""`` when
        the retrieval surfaced nothing at all (which the grounding prompt
        turns into an honest "I don't know", not a fabrication).
    """
    sections: list[str] = []
    used = 0
    facts = facts_block(retrieval.evidence)
    if facts:
        section = _truncate(f"{_GRAPH_FACTS_HEADER}\n{facts}", max_chars)
        if section:
            sections.append(section)
            used = len(section)
    room = max_chars - used - (len(_SECTION_SEPARATOR) if sections else 0)
    blocks = transcript_blocks(retrieval.excerpts, max_chars=max(0, room))
    if blocks:
        sections.append(blocks)
    return _SECTION_SEPARATOR.join(sections)


def synthesize(llm: LLMClient, question: str, context: str) -> str:
    """Write a grounded answer over one retrieval's context.

    Args:
        llm: The chat client (:class:`~varagity.models.llm.LLMClient`, reused
            rather than a bespoke HTTP call, so the app's retry, clamp, and
            context-window discipline all apply).
        question: The question, verbatim — never the condensed or otherwise
            rewritten form (the repo-wide rule: the original words drive the
            answer prompt).
        context: The rendered grounding context, from
            :func:`synthesis_context`.

    Returns:
        The ``<think>``-stripped answer; :data:`NO_EVIDENCE_ANSWER` when the
        context is empty (no model call), or ``""`` when the call failed —
        a scored miss, never a raise.
    """
    if not context.strip():
        return NO_EVIDENCE_ANSWER
    prompt = SYNTHESIS_PROMPT.format(context=context, question=question)
    try:
        raw = llm.generate(
            [{"role": "user", "content": prompt}], max_tokens=SYNTHESIS_MAX_TOKENS, verbose=0
        )
    except Exception:  # a failed synthesis is a scored miss, not a dead run
        logger.warning("graph answer synthesis failed", exc_info=True)
        return ""
    # Mandatory: generate() returns reasoning stages verbatim (the condense and
    # HyDE precedents), and an unstripped block would be scored as the answer.
    return clean_response(raw)


def _truncate(text: str, limit: int) -> str:
    """Cut text to a character limit, marking that something was dropped.

    Args:
        text: The text to fit.
        limit: Maximum characters (a non-positive limit yields ``""``).

    Returns:
        The text, or its head plus an elision marker, never longer than
        ``limit``.
    """
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= len(_ELLIPSIS):
        return text[:limit]
    return text[: limit - len(_ELLIPSIS)].rstrip() + _ELLIPSIS
