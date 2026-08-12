-- Deletions did not travel, and worse, they came back. Measured on two throwaway
-- databases: A deletes a row and re-exports; B imports the bundle without it and
-- keeps the row, because import runs no DELETE at all; B exports its own bundle and
-- A gets the row back. A delete was not slow to propagate — it was undone.
--
-- A tombstone carries identity and nothing else. It must never carry content: a row
-- erased by cuba_forget for GDPR reasons cannot be allowed back in through the
-- record of its own erasure.
--
-- The trigger is AFTER DELETE per row rather than something each delete path
-- remembers to call, and that is also what makes cascades safe to reason about:
-- deleting one entity cascades to its observations, episodes, relations and aliases,
-- and each of those fires its own trigger. So the tombstone set names every row that
-- actually went, one by one. The receiving side can then delete exactly what was
-- named and nothing else, which is the rule that keeps a tombstone for an entity
-- with 3 children here from taking 332 children there.

CREATE TABLE IF NOT EXISTS brain_tombstones (
    table_name  text        NOT NULL,
    row_id      uuid        NOT NULL,
    deleted_at  timestamptz NOT NULL DEFAULT NOW(),
    origin_node text        DEFAULT NULLIF(current_setting('cuba.node_name', true), ''),
    PRIMARY KEY (table_name, row_id)
);

CREATE INDEX IF NOT EXISTS idx_tombstones_deleted_at
    ON brain_tombstones (deleted_at DESC);

CREATE OR REPLACE FUNCTION brain_record_tombstone()
RETURNS TRIGGER AS $func$
BEGIN
    INSERT INTO brain_tombstones (table_name, row_id)
    VALUES (TG_TABLE_NAME, OLD.id)
    ON CONFLICT (table_name, row_id) DO UPDATE SET deleted_at = NOW();
    RETURN OLD;
END;
$func$ LANGUAGE plpgsql;

DO $tombstones$
DECLARE
    t text;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'brain_entities', 'brain_observations', 'brain_episodes',
        'brain_errors', 'brain_relations', 'brain_projects'
    ] LOOP
        EXECUTE format(
            'CREATE TRIGGER %I AFTER DELETE ON %I
             FOR EACH ROW EXECUTE FUNCTION brain_record_tombstone()',
            t || '_tombstone', t
        );
    END LOOP;
END
$tombstones$;

COMMENT ON TABLE brain_tombstones IS
    'Identity of deleted rows, so a delete on one machine reaches the other. Carries no content by design: a row erased for GDPR reasons must not survive inside the record of its erasure. Retention has to be at least as long as the longest a peer can be offline — purge a tombstone before the peer sees it and the row comes back.';
