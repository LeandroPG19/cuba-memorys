CREATE OR REPLACE FUNCTION brain_bump_sync_clock()
RETURNS TRIGGER AS $func$
BEGIN
    NEW.updated_at := NOW();
    NEW.version := OLD.version + 1;
    RETURN NEW;
END;
$func$ LANGUAGE plpgsql;
