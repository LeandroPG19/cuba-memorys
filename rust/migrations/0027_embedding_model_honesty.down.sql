-- Restores the DEFAULT from migration 0007. The nulled-out provenance values
-- are not restored: they asserted a model that never produced them.
ALTER TABLE brain_observations ALTER COLUMN embedding_model SET DEFAULT 'multilingual-e5-small';
ALTER TABLE brain_episodes ALTER COLUMN embedding_model SET DEFAULT 'multilingual-e5-small';
