-- Revert 0035: drop relation provenance.
ALTER TABLE brain_relations DROP CONSTRAINT IF EXISTS brain_relations_provenance_check;
ALTER TABLE brain_relations DROP COLUMN IF EXISTS provenance;
