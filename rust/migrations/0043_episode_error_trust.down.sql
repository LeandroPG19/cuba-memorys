DROP INDEX IF EXISTS idx_errors_quarantined;
DROP INDEX IF EXISTS idx_episodes_quarantined;
ALTER TABLE brain_errors DROP CONSTRAINT IF EXISTS brain_errors_trust_check;
ALTER TABLE brain_errors DROP COLUMN IF EXISTS trust;
ALTER TABLE brain_episodes DROP CONSTRAINT IF EXISTS brain_episodes_trust_check;
ALTER TABLE brain_episodes DROP COLUMN IF EXISTS trust;
