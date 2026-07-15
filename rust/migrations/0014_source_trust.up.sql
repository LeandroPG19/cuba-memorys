-- V0.9: Source credibility tracking (Yin-Han-Yu IEEE TKDE 2008).
--
-- Each observation has a `source` enum (agent/user/inference/...). Without
-- credibility tracking they all weight the same. With this table we keep a
-- per-source Beta(α, β) posterior that gets updated as cuba_calibrar resolves
-- verify_log entries. Sources whose observations turn out wrong over time
-- get down-weighted in cuba_faro scoring.
--
-- α and β start at 1.0 (Beta(1,1) = uniform prior, total ignorance).
-- After 100 outcomes (50 correct, 50 incorrect) we have Beta(51, 51) →
-- p_correct ≈ 0.5 with tight credible interval.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'brain_source_trust'
    ) THEN
        CREATE TABLE brain_source_trust (
            source TEXT PRIMARY KEY,
            alpha FLOAT NOT NULL DEFAULT 1.0 CHECK (alpha >= 1.0),
            beta FLOAT NOT NULL DEFAULT 1.0 CHECK (beta >= 1.0),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );

        -- Pre-seed all valid sources so cuba_faro JOINs always find a row.
        INSERT INTO brain_source_trust (source) VALUES
            ('agent'),
            ('error_detection'),
            ('user'),
            ('consolidation'),
            ('inference')
        ON CONFLICT (source) DO NOTHING;
    END IF;
END $$;
