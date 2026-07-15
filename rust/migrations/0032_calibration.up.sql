-- 0032: persisted calibration constants.
--
-- The OOD abstention threshold cannot be derived from theory on this corpus (see
-- search/calibrate.rs: the embeddings are L2-normalized so they are not Gaussian,
-- and Σ is fitted from ~1400 samples in 384 dimensions). It has to be measured
-- against real queries with `cuba-memorys calibrate`, and the measurement has to
-- survive a restart — otherwise every process re-derives a number that does not
-- work and abstains on everything, which is exactly what was happening.
--
-- Generic key/value on purpose: the next constant that turns out to be corpus-
-- dependent rather than universal goes here too, instead of becoming another
-- hard-coded literal nobody re-measures.

CREATE TABLE IF NOT EXISTS brain_calibration (
    key        TEXT PRIMARY KEY,
    value      DOUBLE PRECISION NOT NULL,
    -- Enough context to know whether the value still applies: the embedding
    -- dimension it was calibrated for, the α it guarantees, the sample sizes.
    -- A threshold calibrated for 384-d says nothing about a 1024-d corpus.
    metadata   JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE brain_calibration IS
    'Corpus-dependent constants measured rather than assumed. See cuba-memorys calibrate.';
