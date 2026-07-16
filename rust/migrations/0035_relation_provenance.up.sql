-- 0035: relation provenance — was this edge typed by hand, or guessed?
--
-- cuba_puente already draws a real line between the two: `create` persists an edge the
-- caller asserted explicitly, while `predict` (Adamic-Adar) and `infer` (transitive
-- closure) only ever returned suggestions — nothing they computed was written back.
-- That line was invisible in the schema: every row in brain_relations looked equally
-- authoritative whether a human typed it or a link-prediction heuristic guessed it.
--
-- 'extracted' is the default for the existing write path (`create`) and for every row
-- that predates this column. 'predicted' is for cuba_puente predict when it is asked to
-- persist its Adamic-Adar suggestions instead of only returning them. 'inferred' is
-- reserved for a future write path off `infer`'s transitive closure — not written yet,
-- but the vocabulary needs to exist before the caller can filter on it.

ALTER TABLE brain_relations
    ADD COLUMN IF NOT EXISTS provenance text NOT NULL DEFAULT 'extracted';

ALTER TABLE brain_relations
    ADD CONSTRAINT brain_relations_provenance_check
    CHECK (provenance IN ('extracted', 'inferred', 'predicted'));

COMMENT ON COLUMN brain_relations.provenance IS
    'extracted = asserted via cuba_puente create; predicted = persisted Adamic-Adar suggestion; inferred = reserved for a future transitive-closure write path.';
