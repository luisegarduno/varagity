-- v3 chat-engine provenance, part 2 of 2 (spec_v3 §8): which chat engine
-- produced an assistant turn (e.g. 'simple' | 'condense_context').
-- Landed inert; nullable, snapshot semantics — see 003.

ALTER TABLE messages ADD COLUMN IF NOT EXISTS chat_engine TEXT;
