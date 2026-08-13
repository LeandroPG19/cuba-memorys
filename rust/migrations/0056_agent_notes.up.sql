-- Two AIs work this database at once and neither can tell the other anything.
-- The peer inbox from 0050 is for the other MACHINE — its own migration says the
-- table is the one thing a remote peer token may write, and its columns are node
-- identity and bundle hashes, which mean nothing between two clients of the same
-- daemon.
--
-- The inbox for local agents already exists and nobody noticed: brain_triggers
-- with condition_type='on_session_start'. It holds a message, it is delivered by
-- cuba_jornada start (handlers/jornada.rs, five lines above where peer notices are
-- injected), it closes itself through max_fires — which is exactly the resolved_at
-- that 0050 needed and that a local note has no bundle to trigger — and it expires.
-- Building a second table would have been a copy of this one with a different name,
-- and a new MCP tool to write it would cost about 175 tokens in the context of
-- every model on every request.
--
-- So this migration only closes the two things that table is missing for the job.
--
-- One: who it is from. Without it the message has to carry its own signature in
-- prose, which is the kind of convention that survives exactly until somebody
-- forgets.
ALTER TABLE brain_triggers ADD COLUMN IF NOT EXISTS from_agent text;

COMMENT ON COLUMN brain_triggers.from_agent IS
    'Which agent left this, for notes between clients of the same daemon. NULL for the reminders a session leaves itself, which is what this table was built for.';

-- Two: a ceiling. Every other inbox in this schema has one (brain_peer_notices at
-- 200 per node, brain_handler_failures at 1000) for the same reason: a table that
-- one caller can write is a table that grows fastest exactly when something is
-- looping, which is the worst moment to also run out of disk. Session-start
-- triggers are the only kind reachable this way, so the cap is scoped to them and
-- leaves the entity reminders alone.
CREATE OR REPLACE FUNCTION brain_cap_agent_notes() RETURNS trigger AS $$
BEGIN
    IF NEW.condition_type <> 'on_session_start' THEN
        RETURN NULL;
    END IF;
    DELETE FROM brain_triggers
    WHERE id IN (
        SELECT id FROM brain_triggers
        WHERE condition_type = 'on_session_start'
        ORDER BY created_at DESC
        OFFSET 200
    );
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER brain_agent_notes_cap
    AFTER INSERT ON brain_triggers
    FOR EACH ROW EXECUTE FUNCTION brain_cap_agent_notes();

-- What is NOT closed here, because it is a property of the delivery and not of the
-- schema: a note is addressed by `similarity(entity_pattern, session_name) > 0.5`.
-- Measured on the two session names in use today the score is 0.29, so they do not
-- cross — but two sessions named alike would deliver each other's notes, and
-- silently. Whoever relies on this for something that matters should address by
-- exact name.
