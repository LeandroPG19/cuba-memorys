CREATE TABLE IF NOT EXISTS brain_embedding_stats (
    stat_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES brain_projects(id) ON DELETE CASCADE,
    dimension INT NOT NULL,
    mean_vector FLOAT[],
    covariance_diag FLOAT[],
    sample_count BIGINT DEFAULT 0,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (project_id, dimension)
);

CREATE OR REPLACE FUNCTION update_calibration_extended(
    p_source_type TEXT,
    p_is_correct BOOLEAN
) RETURNS VOID LANGUAGE plpgsql AS $$
DECLARE
    v_alpha FLOAT;
    v_beta FLOAT;
BEGIN
    SELECT alpha, beta INTO v_alpha, v_beta
    FROM brain_source_trust
    WHERE source = p_source_type;

    IF NOT FOUND THEN
        v_alpha := 1.0;
        v_beta := 1.0;
        INSERT INTO brain_source_trust (source, alpha, beta) VALUES (p_source_type, v_alpha, v_beta);
    END IF;

    IF p_is_correct THEN
        v_alpha := v_alpha + 1.0;
    ELSE
        v_beta := v_beta + 1.0;
    END IF;

    UPDATE brain_source_trust
    SET alpha = v_alpha, beta = v_beta, updated_at = NOW()
    WHERE source = p_source_type;
END;
$$;