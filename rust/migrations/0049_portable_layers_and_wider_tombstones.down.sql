-- The two new triggers go first: the narrow body below cannot serve brain_facts, and
-- leaving one behind would make every DELETE on that table raise 42703.
--
-- brain_record_tombstone() is then put back to the body 0045 shipped, byte for byte,
-- rather than left generalised. The six triggers from 0045 point at this function by
-- oid, so whatever body is here is the body they run; a down migration that leaves a
-- newer one behind means the database no longer matches the migrations it claims.

DROP TRIGGER IF EXISTS brain_facts_tombstone ON brain_facts;
DROP TRIGGER IF EXISTS brain_procedures_tombstone ON brain_procedures;

CREATE OR REPLACE FUNCTION brain_record_tombstone()
RETURNS TRIGGER AS $func$
BEGIN
    INSERT INTO brain_tombstones (table_name, row_id)
    VALUES (TG_TABLE_NAME, OLD.id)
    ON CONFLICT (table_name, row_id) DO UPDATE SET deleted_at = NOW();
    RETURN OLD;
END;
$func$ LANGUAGE plpgsql;

DROP FUNCTION IF EXISTS brain_layer_by_name(text);
