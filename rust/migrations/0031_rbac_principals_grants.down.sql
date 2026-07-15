-- Reverse 0031. Drops the policies first: the function cannot go while a policy
-- still calls it.
DO $$
DECLARE t TEXT;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'brain_entities', 'brain_observations', 'brain_episodes',
        'brain_errors', 'brain_relations'
    ] LOOP
        IF EXISTS (SELECT 1 FROM information_schema.tables
                   WHERE table_schema = 'public' AND table_name = t) THEN
            EXECUTE format('DROP POLICY IF EXISTS principal_grants ON %I', t);
        END IF;
    END LOOP;
END $$;

DROP FUNCTION IF EXISTS brain_principal_can(UUID);
DROP TABLE IF EXISTS brain_grants;
DROP TABLE IF EXISTS brain_principals;
