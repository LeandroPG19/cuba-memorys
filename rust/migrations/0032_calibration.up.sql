
CREATE TABLE IF NOT EXISTS brain_calibration (
    key        TEXT PRIMARY KEY,
    value      DOUBLE PRECISION NOT NULL,
    metadata   JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE brain_calibration IS
    'Corpus-dependent constants measured rather than assumed. See cuba-memorys calibrate.';
