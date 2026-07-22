-- Sidebar conversation groups — user-created folders over the conversation
-- list. Conversations join a group via a nullable FK; deleting a group
-- detaches its conversations (ON DELETE SET NULL) rather than deleting
-- them, so a folder is organization, never ownership.

CREATE TABLE IF NOT EXISTS conversation_groups (
    group_id    TEXT PRIMARY KEY,        -- app-generated id
    name        TEXT NOT NULL,           -- display name (not unique — ids are the identity)
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE conversations
    ADD COLUMN IF NOT EXISTS group_id TEXT
        REFERENCES conversation_groups(group_id) ON DELETE SET NULL;

-- The sidebar partitions the list by group.
CREATE INDEX IF NOT EXISTS conversations_group_id_idx ON conversations(group_id);
