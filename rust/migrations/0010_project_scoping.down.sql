DO $$
DECLARE t TEXT;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'brain_entities','brain_observations',
        'brain_episodes','brain_sessions',
        'brain_errors','brain_relations'
    ]
    LOOP
        EXECUTE format('DROP INDEX IF EXISTS idx_%s_project', t);
        EXECUTE format('ALTER TABLE %I DROP COLUMN IF EXISTS project_id', t);
    END LOOP;
END $$;
DROP TABLE IF EXISTS brain_projects CASCADE;
