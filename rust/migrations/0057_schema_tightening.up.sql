-- Two gaps an audit found by running the schema rather than reading it. Neither is
-- reachable today; both are the kind of thing that becomes reachable the moment
-- somebody writes a new caller and assumes the column already guaranteed it.
--
-- One: brain_tombstones.table_name is free text. `INSERT INTO brain_tombstones
-- (table_name, row_id) VALUES ('pg_authid', ...)` is accepted by the schema. The
-- code refuses it — apply_tombstones bails on any table outside TOMBSTONED_TABLES,
-- and that check is what actually protects the database — so this is defence in
-- depth, not a hole being closed. It is worth having because a tombstone is a
-- licence to delete rows, and the list of tables that may be deleted from should
-- be true in two places, not one.
--
-- The eight names are the same eight the trigger loop of 0045 and 0049 installs,
-- and the same eight TOMBSTONED_TABLES carries. A test already asserts those two
-- agree; this makes it three.
DELETE FROM brain_tombstones
WHERE table_name NOT IN (
    'brain_entities', 'brain_observations', 'brain_episodes', 'brain_errors',
    'brain_relations', 'brain_projects', 'brain_procedures', 'brain_facts'
);

ALTER TABLE brain_tombstones
    ADD CONSTRAINT brain_tombstones_known_table
    CHECK (table_name IN (
        'brain_entities', 'brain_observations', 'brain_episodes', 'brain_errors',
        'brain_relations', 'brain_projects', 'brain_procedures', 'brain_facts'
    ));

-- Two: brain_observations.version is nullable, and brain_bump_sync_clock does
-- `NEW.version := OLD.version + 1`. A row that ever held NULL there would keep the
-- logical clock dead forever — every later edit would compute NULL + 1 and store
-- NULL again, so the row would look unchanged to every other machine no matter how
-- many times it was corrected. Measured on the live corpus: 0 NULLs in 1888 rows,
-- and the import already writes COALESCE(version, 1), so nothing can produce one
-- today. The backfill runs first anyway, because SET NOT NULL scans the table and
-- an install with one stray NULL would otherwise fail to start.
UPDATE brain_observations SET version = 1 WHERE version IS NULL;
ALTER TABLE brain_observations ALTER COLUMN version SET NOT NULL;
