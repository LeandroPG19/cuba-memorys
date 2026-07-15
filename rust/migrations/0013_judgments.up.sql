DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'brain_judgments'
    ) THEN
        CREATE TABLE brain_judgments (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            observation_a UUID REFERENCES brain_observations(id) ON DELETE CASCADE,
            observation_b UUID REFERENCES brain_observations(id) ON DELETE CASCADE,
            verdict TEXT NOT NULL CHECK (verdict IN
                ('contradicts','supersedes','complementary','unrelated','unknown')),
            confidence FLOAT NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
            reason TEXT,
            judge_backend TEXT NOT NULL,
            judge_model TEXT,
            project_id UUID REFERENCES brain_projects(id) ON DELETE SET NULL,
            judged_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE (observation_a, observation_b)
        );
        CREATE INDEX idx_judgments_pair
            ON brain_judgments(observation_a, observation_b);
        CREATE INDEX idx_judgments_project
            ON brain_judgments(project_id) WHERE project_id IS NOT NULL;
        CREATE INDEX idx_judgments_verdict
            ON brain_judgments(verdict)
            WHERE verdict IN ('contradicts','supersedes');
    END IF;
END $$;
