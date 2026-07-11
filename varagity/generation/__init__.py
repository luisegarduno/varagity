"""Answer generation: context prompt construction and grounded generation.

The query pipeline's final stage (spec §10.2) — and, via
:func:`~varagity.generation.answer.answer_query`, the whole
retrieve → format → generate thread (spec §10.1).
"""

from varagity.generation.answer import (
    ANSWER_PROMPT,
    QueryState,
    answer_query,
    format_context,
    generate_answer,
)

__all__ = [
    "ANSWER_PROMPT",
    "QueryState",
    "answer_query",
    "format_context",
    "generate_answer",
]
