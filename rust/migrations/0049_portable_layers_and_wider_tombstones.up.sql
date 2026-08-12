-- Two things had to exist before brain_facts, brain_procedures and brain_source_trust
-- could travel in a bundle at all.
--
-- THE FIRST IS AN IMPORT KILLER, and it is not hypothetical. brain_facts.layer_id is a
-- FK to brain_memory_layers(layer_id), which is gen_random_uuid(), and migration 0020
-- inserts the four layers without pinning an id. So the ids are per installation.
-- Measured on two databases migrated from the same 48 files in the same container:
-- 0 of 4 layer ids coincide (episodic here is 990e8f5c…, there 0db1cad8…). Importing
-- one machine's facts into another therefore raises 23503, and because import runs in
-- one transaction the failure is not "one fact skipped": the next statement gets
-- "current transaction is aborted", the COMMIT turns into a ROLLBACK and the whole
-- bundle lands zero rows. Reproduced: two INSERTs, the second one perfectly valid,
-- 0 rows after.
--
-- The fix is to make the bundle carry the layer NAME, which is the same four labels on
-- every install, and resolve it to the local id on the way in. This lookup is a
-- function and not an inline subselect for one reason worth a line: the obvious
-- resolution, layer_id = (SELECT … WHERE layer_name = $1::memory_layer_type), raises
-- 22P02 on any text that is not one of the four enum labels — which is the same
-- transaction-wide abort we came here to remove, just with a different SQLSTATE. So
-- the comparison is on layer_name::text and an unknown name resolves to NULL. NULL is
-- the honest answer, not a silent downgrade: layer_id is nullable, all 990 facts in
-- the live corpus have it NULL today, and a fact that arrives with no layer is a fact,
-- while a fact that aborts the import costs the other 989.
--
-- THE SECOND is that brain_record_tombstone() reads OLD.id, and brain_facts has no id
-- column — its key is fact_id. A trigger with that body on brain_facts fails at DELETE
-- time with 42703, which would turn "a delete travels" into "a delete is impossible".
-- Rather than a second near-identical function, the existing one now reads its key out
-- of to_jsonb(OLD) using a column name passed as a trigger argument, defaulting to
-- 'id'. The six triggers from 0045 are declared with no arguments, so TG_ARGV[0] is
-- NULL for them and they keep meaning exactly what they meant.
--
-- A key that resolves to NULL raises instead of returning quietly. The only way to get
-- there is a trigger declared with the wrong column name, and a tombstone that is not
-- written is a deletion that comes back on the next sync — the precise bug 0045 was
-- written to kill. Better to fail on the first DELETE than to lose deletes forever.
--
-- brain_source_trust is deliberately NOT given a trigger. Its primary key is
-- `source text` ('cli', 'inference', …) and brain_tombstones.row_id is uuid NOT NULL,
-- so there is nothing to write there without widening that column for every table.
-- It does not need one either: the table is a Beta posterior per source, five rows
-- here, and a source that stops being trusted is updated, not deleted.

CREATE OR REPLACE FUNCTION brain_layer_by_name(p_name text)
RETURNS uuid
LANGUAGE sql
STABLE
AS $$
    SELECT layer_id
    FROM brain_memory_layers
    WHERE layer_name::text = p_name;
$$;

COMMENT ON FUNCTION brain_layer_by_name(text) IS
    'Resolves a memory layer name to this installation''s layer_id, or NULL if the name is not one of the four. The four ids are gen_random_uuid() per install (measured: 0 of 4 shared between two databases built from the same migrations), so a bundle has to carry the name and resolve it here. The comparison is on layer_name::text on purpose: casting unknown text to memory_layer_type raises 22P02 and aborts the importing transaction, which is the failure this exists to prevent.';

CREATE OR REPLACE FUNCTION brain_record_tombstone()
RETURNS TRIGGER AS $func$
DECLARE
    key_column text := COALESCE(TG_ARGV[0], 'id');
    key_value  text := to_jsonb(OLD) ->> key_column;
BEGIN
    IF key_value IS NULL THEN
        RAISE EXCEPTION
            'tombstone trigger on % names key column %, which that table does not have',
            TG_TABLE_NAME, key_column;
    END IF;
    INSERT INTO brain_tombstones (table_name, row_id)
    VALUES (TG_TABLE_NAME, key_value::uuid)
    ON CONFLICT (table_name, row_id) DO UPDATE SET deleted_at = NOW();
    RETURN OLD;
END;
$func$ LANGUAGE plpgsql;

CREATE TRIGGER brain_procedures_tombstone
    AFTER DELETE ON brain_procedures
    FOR EACH ROW EXECUTE FUNCTION brain_record_tombstone();

CREATE TRIGGER brain_facts_tombstone
    AFTER DELETE ON brain_facts
    FOR EACH ROW EXECUTE FUNCTION brain_record_tombstone('fact_id');
