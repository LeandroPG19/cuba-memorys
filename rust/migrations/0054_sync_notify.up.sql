-- The clock from 0047 already knows the thing that is hard to know: which
-- updates matter to another machine. Its trigger fires BEFORE UPDATE OF content,
-- observation_type, trust, evidence, tags, so decay — which moves importance on
-- 1097 rows every four hours — and reembed, which moves only the vector, never
-- reach it. That distinction took a refutation to get right and is not being
-- rebuilt here: this migration only makes the same function say out loud what it
-- already decided.
--
-- 0047 cannot be edited. sqlx hashes every applied migration and changing one
-- stops every existing database from starting — that happened in e96df5d and was
-- undone in a245f62. CREATE OR REPLACE FUNCTION reaches the same function by
-- name without touching that file, and the trigger 0047 created keeps pointing
-- at it.
--
-- The payload is an identifier and nothing else. NOTIFY has a hard 8000-byte
-- limit that no setting moves, and an observation's content routinely exceeds
-- it; sending the text would turn an ordinary write into a failed transaction.
-- The listener does not need it either: it wakes up, and the peer comes and
-- takes the rows through the path that already validates them.
--
-- What this does NOT give you: durability. If nobody is listening when it fires,
-- the event is gone for good. That is why the periodic fetch stays — NOTIFY only
-- removes the waiting, it is not the channel.

CREATE OR REPLACE FUNCTION brain_bump_sync_clock()
RETURNS TRIGGER AS $func$
BEGIN
    NEW.updated_at := NOW();
    NEW.version := OLD.version + 1;
    PERFORM pg_notify('brain_sync_clock', TG_TABLE_NAME || ':' || NEW.id::text);
    RETURN NEW;
END;
$func$ LANGUAGE plpgsql;
