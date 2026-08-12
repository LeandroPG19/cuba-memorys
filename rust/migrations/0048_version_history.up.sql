-- Two holes in "keep both versions, never lose one".
--
-- The first: sync's import rewrote `content` without touching `previous_versions` at
-- all, so `--conflict overwrite` replaced a correction made on this machine with the
-- peer's version and left no trace that anything had been there. The whole promise
-- of the conflict design rests on the loser being kept, and the one path most likely
-- to hit a conflict did not keep it.
--
-- The second: `previous_versions` had no bound. Every conflict appends, the array is
-- read back on every retrieval, and a disputed row would grow until it is the most
-- expensive thing in the graph. "Keep both" without a cap is a leak wearing a
-- promise's clothes.
--
-- One function, used by both eco's correct() and sync's import, so the two cannot
-- drift into different ideas of what history looks like.

CREATE OR REPLACE FUNCTION brain_append_version(
    history jsonb,
    entry   jsonb,
    cap     integer DEFAULT 20
) RETURNS jsonb LANGUAGE sql IMMUTABLE AS $$
    SELECT COALESCE(
        (
            SELECT jsonb_agg(e ORDER BY i)
            FROM (
                SELECT e, i
                FROM jsonb_array_elements(COALESCE(history, '[]'::jsonb) || entry)
                     WITH ORDINALITY AS t(e, i)
                ORDER BY i DESC
                LIMIT GREATEST(cap, 1)
            ) kept
        ),
        '[]'::jsonb
    );
$$;

COMMENT ON FUNCTION brain_append_version(jsonb, jsonb, integer) IS
    'Appends one superseded version and keeps only the newest `cap`. Both the correction path and the sync import go through here so the two cannot drift; the cap exists because previous_versions is read on every retrieval and an unbounded array turns a disputed row into the slowest one in the graph.';
