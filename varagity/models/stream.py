"""Streaming ``<think>`` splitter — the token-stream twin of ``clean_response``.

:func:`varagity.models.llm.clean_response` strips reasoning stages from a
*complete* response; the SSE chat path (spec_v2 §4.3) needs the same
classification while tokens are still arriving, so reasoning deltas stream
as ``reasoning`` events and everything else as ``token`` events.
:class:`ThinkStreamSplitter` is that state machine. It handles the shapes
``clean_response`` hardens against:

* balanced ``<think>…</think>`` blocks (however many), with tags split
  across token boundaries;
* an unclosed ``<think>`` (generation hit the token cap mid-reasoning) —
  everything after the opener classifies as reasoning;
* an orphaned ``</think>`` (some chat templates consume the opening tag).

The orphan case is the one genuinely streaming-hostile shape: the text
before the closer *was* reasoning, but a splitter cannot reclassify deltas
it already emitted as ``answer``. It flags the stream instead
(:attr:`ThinkStreamSplitter.saw_orphan_closer`), and callers that persist
the answer reconcile by running ``clean_response`` over the accumulated raw
text — the streaming display is best-effort, the stored answer is exact.
"""

from typing import Literal

Kind = Literal["reasoning", "answer"]
"""Classification of one streamed text fragment."""

_OPEN_TAG = "<think>"
_CLOSE_TAG = "</think>"


def _partial_tag_suffix(text: str, tags: tuple[str, ...]) -> int:
    """Length of the longest ``text`` suffix that could begin one of ``tags``.

    Backs the splitter's hold-back: a delta ending in ``"<thi"`` must not be
    emitted yet, because the next delta may complete a tag.

    Args:
        text: The pending, not-yet-classified text.
        tags: Candidate tags (only proper prefixes count — a complete tag is
            handled by the scan, not the hold-back).

    Returns:
        The hold-back length; ``0`` when the text cannot end mid-tag.
    """
    best = 0
    for tag in tags:
        for length in range(min(len(text), len(tag) - 1), 0, -1):
            if text[-length:] == tag[:length]:
                best = max(best, length)
                break
    return best


class ThinkStreamSplitter:
    """Classify streamed LLM deltas as ``reasoning`` or ``answer`` text.

    Feed raw deltas in arrival order with :meth:`feed`; each call returns the
    fragments that became classifiable (tag characters themselves are never
    emitted). Call :meth:`finalize` once at end-of-stream to flush any held-
    back partial tag.

    Attributes:
        saw_orphan_closer: ``True`` once a ``</think>`` arrived with no open
            ``<think>`` block — the already-emitted ``answer`` fragments were
            actually reasoning. Callers needing exact ``clean_response``
            semantics re-clean the accumulated raw text when set.
    """

    def __init__(self) -> None:
        """Start in the answer state with nothing pending."""
        self._pending = ""
        self._in_reasoning = False
        self.saw_orphan_closer = False

    def feed(self, delta: str) -> list[tuple[Kind, str]]:
        """Consume one raw delta and return the newly classifiable fragments.

        Args:
            delta: The next raw text fragment from the LLM stream (may be
                empty; may split a tag anywhere).

        Returns:
            ``(kind, text)`` tuples in stream order; possibly empty while the
            splitter holds back a potential partial tag.
        """
        self._pending += delta
        events: list[tuple[Kind, str]] = []
        while self._pending:
            if self._in_reasoning:
                if not self._scan_reasoning(events):
                    break
            elif not self._scan_answer(events):
                break
        return events

    def finalize(self) -> list[tuple[Kind, str]]:
        """Flush the hold-back at end-of-stream.

        A pending fragment can no longer become a tag once the stream ends:
        it classifies under the current state (reasoning inside an unclosed
        ``<think>`` block, answer otherwise).

        Returns:
            At most one ``(kind, text)`` tuple.
        """
        if not self._pending:
            return []
        kind: Kind = "reasoning" if self._in_reasoning else "answer"
        events: list[tuple[Kind, str]] = [(kind, self._pending)]
        self._pending = ""
        return events

    def _scan_reasoning(self, events: list[tuple[Kind, str]]) -> bool:
        """Consume pending text inside a ``<think>`` block.

        Only ``</think>`` is meaningful here — a nested ``<think>`` is inert
        reasoning text, matching ``clean_response``'s non-greedy block regex.

        Args:
            events: Output accumulator to append classified fragments to.

        Returns:
            ``True`` if a state transition consumed a tag (scan again);
            ``False`` when the pending text is exhausted for this call.
        """
        index = self._pending.find(_CLOSE_TAG)
        if index != -1:
            if index:
                events.append(("reasoning", self._pending[:index]))
            self._pending = self._pending[index + len(_CLOSE_TAG) :]
            self._in_reasoning = False
            return True
        emit_up_to = len(self._pending) - _partial_tag_suffix(self._pending, (_CLOSE_TAG,))
        if emit_up_to:
            events.append(("reasoning", self._pending[:emit_up_to]))
            self._pending = self._pending[emit_up_to:]
        return False

    def _scan_answer(self, events: list[tuple[Kind, str]]) -> bool:
        """Consume pending text outside any ``<think>`` block.

        Both tags are meaningful here: ``<think>`` opens a reasoning block;
        an orphaned ``</think>`` is dropped and flagged (see the module
        docstring on why it cannot be reclassified retroactively).

        Args:
            events: Output accumulator to append classified fragments to.

        Returns:
            ``True`` if a state transition consumed a tag (scan again);
            ``False`` when the pending text is exhausted for this call.
        """
        matches = [
            (index, tag)
            for tag in (_OPEN_TAG, _CLOSE_TAG)
            if (index := self._pending.find(tag)) != -1
        ]
        if matches:
            index, tag = min(matches)
            if index:
                events.append(("answer", self._pending[:index]))
            self._pending = self._pending[index + len(tag) :]
            if tag == _OPEN_TAG:
                self._in_reasoning = True
            else:
                self.saw_orphan_closer = True
            return True
        emit_up_to = len(self._pending) - _partial_tag_suffix(
            self._pending, (_OPEN_TAG, _CLOSE_TAG)
        )
        if emit_up_to:
            events.append(("answer", self._pending[:emit_up_to]))
            self._pending = self._pending[emit_up_to:]
        return False
