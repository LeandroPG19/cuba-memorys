
ALTER TABLE brain_observations ADD COLUMN IF NOT EXISTS last_decayed_at TIMESTAMPTZ;
UPDATE brain_observations SET last_decayed_at = NOW() WHERE last_decayed_at IS NULL;
ALTER TABLE brain_observations ALTER COLUMN last_decayed_at SET DEFAULT NOW();
