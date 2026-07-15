
ALTER TABLE brain_observations ALTER COLUMN embedding_model DROP DEFAULT;
ALTER TABLE brain_episodes ALTER COLUMN embedding_model DROP DEFAULT;

UPDATE brain_observations
   SET embedding_model = NULL
 WHERE embedding IS NULL
   AND embedding_model IS NOT NULL;

UPDATE brain_episodes
   SET embedding_model = NULL
 WHERE embedding IS NULL
   AND embedding_model IS NOT NULL;
