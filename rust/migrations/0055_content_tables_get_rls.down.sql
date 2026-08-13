DO $$
DECLARE t text;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'brain_compaction_snapshots',
        'brain_wm',
        'brain_observation_chunks',
        'brain_procedures',
        'brain_judgments',
        'brain_embedding_stats',
        'brain_sync_conflicts'
    ]
    LOOP
        EXECUTE format('DROP POLICY IF EXISTS tenant_isolation ON %I', t);
        EXECUTE format('ALTER TABLE %I NO FORCE ROW LEVEL SECURITY', t);
        EXECUTE format('ALTER TABLE %I DISABLE ROW LEVEL SECURITY', t);
    END LOOP;
END $$;
