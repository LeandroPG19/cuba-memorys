-- Nothing has ever recorded that a tool call failed. The line goes to tracing and
-- from there to the journal, where it survives until the journal rotates and where
-- nobody looks unless they already suspect something. So the honest answer to "has
-- anything been going wrong?" has been: open journalctl and read.
--
-- Only failures land here, and that is the whole design. cuba_faro is called
-- constantly; a row per call would make the telemetry table grow faster than the
-- knowledge it watches, on a corpus that takes four writes a day. Failures are rare
-- and are the only thing anyone wants to read days later. The recent successful
-- calls live in a ring in memory and are gone on restart, which the panel says out
-- loud rather than implying it has the whole history.
--
-- `error` is redacted before it gets here: a connection failure can carry the
-- database URL, password included, straight into the message.

CREATE TABLE IF NOT EXISTS brain_handler_failures (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tool text NOT NULL,
    client text,
    error text NOT NULL,
    elapsed_ms integer NOT NULL,
    created_at timestamptz NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_handler_failures_recent
    ON brain_handler_failures (created_at DESC);

-- Same reasoning as brain_peer_notices: a table written by a failure path is a table
-- that grows fastest exactly when something is looping, which is the worst moment to
-- also run out of disk.
CREATE OR REPLACE FUNCTION brain_cap_handler_failures() RETURNS trigger AS $$
BEGIN
    DELETE FROM brain_handler_failures
    WHERE id IN (
        SELECT id FROM brain_handler_failures
        ORDER BY created_at DESC
        OFFSET 1000
    );
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER brain_handler_failures_cap
    AFTER INSERT ON brain_handler_failures
    FOR EACH ROW EXECUTE FUNCTION brain_cap_handler_failures();

GRANT SELECT, INSERT, UPDATE, DELETE ON brain_handler_failures TO cuba_app;

COMMENT ON TABLE brain_handler_failures IS
    'Tool calls that returned an error, with the message redacted. Successes are not recorded: they live in a 500-entry ring in the daemon and are lost on restart. Capped at 1000 rows by trigger.';
