-- A peer holding CUBA_PEER_TOKEN can read this node's memory and cannot write a
-- row of it. That is the right default, but it leaves the peer mute: the model
-- running on the other machine may have just learned something this one needs,
-- and had no way to say so. Polling is the alternative, and polling is how a
-- system that claims to be live ends up minutes behind.
--
-- This table is the one thing a peer may write, and it is deliberately not
-- memory: no entity, no observation, no embedding, nothing that search or
-- recall will ever return. It is a bell. The local model reads the notice,
-- decides whether it cares, and if it does it pulls from the peer and imports
-- with the code that already validates and quarantines. The peer never decides
-- what enters this database, which is exactly the property the read-only token
-- was chosen for.
--
-- `node_id` is self-asserted: the peer says which node it is, and nothing here
-- proves it. That is honest for two machines sharing one secret — the token is
-- the authentication, the node id is a label — and it is why the column is not
-- a foreign key to anything.

CREATE TABLE IF NOT EXISTS brain_peer_notices (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    node_id uuid,
    node_name text,
    summary text NOT NULL,
    manifest_hash text,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    resolved_at timestamptz
);

CREATE INDEX IF NOT EXISTS idx_peer_notices_pending
    ON brain_peer_notices (created_at DESC)
    WHERE resolved_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_peer_notices_hash
    ON brain_peer_notices (manifest_hash)
    WHERE resolved_at IS NULL;

-- Without a cap, the one write a peer is allowed is also a way to fill the disk
-- of the machine that trusted it. The trigger keeps the most recent notices per
-- node and drops the rest, which is the same reasoning as the tombstone alarm:
-- a bounded blast radius beats a guard nobody can be sure fires.
CREATE OR REPLACE FUNCTION brain_cap_peer_notices() RETURNS trigger AS $$
BEGIN
    DELETE FROM brain_peer_notices
    WHERE id IN (
        SELECT id FROM brain_peer_notices
        WHERE node_id IS NOT DISTINCT FROM NEW.node_id
        ORDER BY created_at DESC
        OFFSET 200
    );
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER brain_peer_notices_cap
    AFTER INSERT ON brain_peer_notices
    FOR EACH ROW EXECUTE FUNCTION brain_cap_peer_notices();

GRANT SELECT, INSERT, UPDATE, DELETE ON brain_peer_notices TO cuba_app;

COMMENT ON TABLE brain_peer_notices IS
    'The only table a CUBA_PEER_TOKEN may write. Not memory: a signal that the other machine has something, for the local model to act on or ignore. Capped at 200 rows per node by trigger.';
