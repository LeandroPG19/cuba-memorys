-- 0027: stop `embedding_model` from lying about rows that were never embedded.
--
-- Migration 0007 added the column as:
--     embedding_model TEXT DEFAULT 'multilingual-e5-small'
--
-- but the embedding itself is written by a *separate* UPDATE after the INSERT
-- (handlers/cronica.rs), and that UPDATE is skipped whenever the real ONNX
-- model is not loaded -- see the guard in cronica.rs, which deliberately
-- prefers NULL over a semantically meaningless hash vector.
--
-- Net effect: every row inserted without an embedding still claimed
-- `embedding_model = 'multilingual-e5-small'`. On the reference database that
-- was 205 of 1420 observations asserting provenance from a model that never
-- touched them. `cuba_zafra action=reembed` selects on
-- `embedding_model != $2 OR embedding_model IS NULL OR embedding IS NULL`,
-- so it still finds them -- but any consumer trusting the column was misled.
--
-- Fix: drop the DEFAULT (the actual root cause -- an INSERT that omits the
-- column now stores NULL) and null out the rows that never had a vector.
--
-- Not enforced as a CHECK yet: `handlers/sync.rs` imports observations with an
-- embedding_model but no embedding (the vectors travel separately in
-- embeddings.bin.zst, and the restoring UPDATE does not set the model column).
-- Tightening that path first is a prerequisite for
--     CHECK ((embedding IS NULL) = (embedding_model IS NULL))

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
