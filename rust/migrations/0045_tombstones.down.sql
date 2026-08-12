DO $drop_tombstones$
DECLARE
    t text;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'brain_entities', 'brain_observations', 'brain_episodes',
        'brain_errors', 'brain_relations', 'brain_projects'
    ] LOOP
        EXECUTE format('DROP TRIGGER IF EXISTS %I ON %I', t || '_tombstone', t);
    END LOOP;
END
$drop_tombstones$;

DROP FUNCTION IF EXISTS brain_record_tombstone();
DROP INDEX IF EXISTS idx_tombstones_deleted_at;
DROP TABLE IF EXISTS brain_tombstones;
