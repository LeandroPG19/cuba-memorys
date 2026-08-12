-- Nothing has ever known what the other machine last handed over. The only loop
-- breaker was brain_sync_state, which deduplicates by manifest hash — and the
-- hash changes whenever anything moves, so the cycle converged in data and
-- never in work: each side kept importing bundles that inserted zero rows.
--
-- One row per peer, holding what came back last. `last_manifest_hash` is the
-- cursor: a fetch that comes back with the same hash has nothing new and stops
-- before opening a transaction. `last_error` is kept because a peer that has
-- been unreachable for two days should say so somewhere a person will look,
-- rather than being indistinguishable from a peer with nothing to send.
--
-- No token here. The secret lives in CUBA_PEER_TOKEN, in the environment, for
-- the same reason no other credential is in this database: a row is backed up,
-- exported and replicated, and this one would then be in a bundle travelling to
-- the very machine it authenticates against.

CREATE TABLE IF NOT EXISTS brain_sync_peers (
    name text PRIMARY KEY,
    url text NOT NULL,
    last_manifest_hash text,
    last_synced_at timestamptz,
    last_rows_inserted integer,
    last_error text,
    created_at timestamptz NOT NULL DEFAULT NOW()
);

GRANT SELECT, INSERT, UPDATE, DELETE ON brain_sync_peers TO cuba_app;

COMMENT ON TABLE brain_sync_peers IS
    'One row per machine this node syncs with. last_manifest_hash is the cursor that lets a fetch stop before it opens a transaction when nothing changed. The bearer token is deliberately not here: it would travel in the bundle it authenticates.';
