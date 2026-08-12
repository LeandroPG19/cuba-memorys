-- The import has reported `diverged` since the merge fix: the ids of rows that
-- exist here with different content from the bundle's copy. That list dies with
-- the tool call. Whoever ran the sync sees a number, and by the next session
-- there is nothing left to look at — so "reported as a divergence", which this
-- plan says in half a dozen places, has meant "printed once and forgotten".
--
-- One row per unresolved disagreement, holding both sides. Both, because the
-- point is that a person can read what each machine says and decide; a conflict
-- record that only keeps the loser's id is a pointer to something already gone.
--
-- `resolution` is written when it is closed and never blanked: 'ours' kept what
-- was already here, 'theirs' took the incoming text and pushed the local one
-- into previous_versions, 'both' kept this row and recorded the other as a
-- version. There is no fourth option that discards anything.

CREATE TABLE IF NOT EXISTS brain_sync_conflicts (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    observation_id uuid NOT NULL REFERENCES brain_observations(id) ON DELETE CASCADE,
    local_content text NOT NULL,
    incoming_content text NOT NULL,
    incoming_origin_node text,
    manifest_hash text,
    detected_at timestamptz NOT NULL DEFAULT NOW(),
    resolved_at timestamptz,
    resolution text CHECK (resolution IN ('ours', 'theirs', 'both'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_sync_conflicts_open
    ON brain_sync_conflicts (observation_id)
    WHERE resolved_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_sync_conflicts_recent
    ON brain_sync_conflicts (detected_at DESC);

ALTER TABLE brain_sync_conflicts
    ADD CONSTRAINT brain_sync_conflicts_closed_says_how
    CHECK ((resolved_at IS NULL) = (resolution IS NULL));

GRANT SELECT, INSERT, UPDATE, DELETE ON brain_sync_conflicts TO cuba_app;

COMMENT ON TABLE brain_sync_conflicts IS
    'Rows two machines disagree about, with both texts kept so a person can read them. Written by import when conflict=merge or skip leaves an incoming version on the floor; closed by cuba_sync action=resolve. The unique partial index means one open conflict per observation: a second import of the same disagreement updates it rather than piling up.';
