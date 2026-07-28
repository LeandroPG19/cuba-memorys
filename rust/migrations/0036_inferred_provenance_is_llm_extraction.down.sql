-- Revert 0036: restore the 0035 wording.
COMMENT ON COLUMN brain_relations.provenance IS
    'extracted = asserted via cuba_puente create; predicted = persisted Adamic-Adar suggestion; inferred = reserved for a future transitive-closure write path.';
