-- The tiebreak for a sync conflict has to name a machine, and until now nothing
-- could. The chain was CUBA_NODE_NAME -> HOSTNAME -> COMPUTERNAME -> empty string,
-- and HOSTNAME is a shell variable that is not exported to child processes: measured,
-- zero matches in the environment of this daemon. The systemd unit ships with
-- CUBA_NODE_NAME commented out. So the daemon writes origin_node = NULL, which is
-- exactly the 240 rows out of 1880 that carry no origin at all.
--
-- And when it is set, nothing stops two machines from picking the same name. `pop-os`
-- on both is the likeliest outcome there is.
--
-- Identity therefore lives in the database, one row per installation, generated once
-- and stable across restarts. CUBA_NODE_NAME stays exactly as it is and keeps meaning
-- what it means: a label a human reads. The uuid is what a machine compares.

CREATE TABLE IF NOT EXISTS brain_node_identity (
    node_id    uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    label      text,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    only_one   boolean     NOT NULL DEFAULT TRUE UNIQUE CHECK (only_one)
);

INSERT INTO brain_node_identity (label)
SELECT NULLIF(current_setting('cuba.node_name', true), '')
WHERE NOT EXISTS (SELECT 1 FROM brain_node_identity);

COMMENT ON TABLE brain_node_identity IS
    'One row, generated on first migration and never again. This is what a sync conflict tiebreak compares, because it is unique by construction; CUBA_NODE_NAME is a human label and two machines can easily share one.';
