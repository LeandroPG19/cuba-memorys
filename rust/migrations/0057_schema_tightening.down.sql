ALTER TABLE brain_observations ALTER COLUMN version DROP NOT NULL;
ALTER TABLE brain_tombstones DROP CONSTRAINT IF EXISTS brain_tombstones_known_table;
