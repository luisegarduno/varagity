"""The condense prompt (spec_v3 §4.5) and its history formatting.

A module-level constant following ``ANSWER_PROMPT``
(:mod:`varagity.generation.answer`) exactly — plain :meth:`str.format`, no
templating engine — sent as a single user message via the non-streaming
:meth:`~varagity.models.llm.LLMClient.generate` path.

The topic-shift instruction is the prompt's load-bearing line: dragging the
old topic into a question that changed the subject is the failure mode that
matters most for a naive condenser (spec_v3 §13.4), and the chat-engine eval
fixtures test for it explicitly.
"""

from collections.abc import Sequence

from varagity.chat.base import Turn

CONDENSE_PROMPT = """Given the conversation below and a follow-up question, rewrite the \
follow-up into ONE standalone search query that names whatever its pronouns and \
references point to.
Keep the follow-up's own words and intent wherever possible. Add nothing it does not \
ask about — if the follow-up changes the subject, do not carry the old topic along.
Reply with the standalone query only — no quotes, no explanation.

<conversation>
{history}
</conversation>

FOLLOW-UP: {query}
STANDALONE QUERY:"""

# The completion-priming label the prompt ends with. Exposed so the engine
# can strip a model's echo of it from the answer (a chat-engine eval
# finding); a test asserts the template still ends with it, so the two
# can't drift apart.
CONDENSE_QUERY_LABEL = "STANDALONE QUERY:"


def format_history(turns: Sequence[Turn]) -> str:
    """Render conversation turns into the prompt's ``<conversation>`` block.

    Args:
        turns: Prior turns, oldest first.

    Returns:
        One ``role: content`` line per turn, in order.
    """
    return "\n".join(f"{turn.role}: {turn.content}" for turn in turns)
