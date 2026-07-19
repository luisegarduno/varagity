# Chat engines

The chat-engine registry (spec_v3 §4;
[ADR-011](../adr/ADR-011-chat-engine-condense.md)): an engine decides
**what string the retriever searches with**, given the turn and its
conversation history, returning the `PreparedQuery` two-string split —
`search_query` drives retrieval while `original_query` (always the user's
words) drives generation. Registered engines: `simple` (the stateless
identity split) and `condense_context` (history-resolved standalone
queries; the shipped default stays `simple` — benchmark-decided).

::: varagity.chat.base

::: varagity.chat.simple

::: varagity.chat.condense

::: varagity.chat.prompts
