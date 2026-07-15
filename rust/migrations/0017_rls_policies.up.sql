DO $$
DECLARE t TEXT;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'brain_entities','brain_observations',
        'brain_episodes','brain_sessions',
        'brain_errors','brain_relations'
    ]
    LOOP
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
        EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', t);

        EXECUTE format('DROP POLICY IF EXISTS tenant_isolation ON %I', t);

        EXECUTE format(
            'CREATE POLICY tenant_isolation ON %I
             USING (
                 current_setting(''app.current_project'', true) IS NULL
              OR current_setting(''app.current_project'', true) = ''''
              OR current_setting(''app.current_project'', true) = ''*''
              OR project_id IS NULL
              OR project_id::text = current_setting(''app.current_project'', true)
             )', t);
    END LOOP;
END $$;
