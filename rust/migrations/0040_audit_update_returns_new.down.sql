CREATE OR REPLACE FUNCTION brain_audit_block_mutation()
RETURNS TRIGGER AS $func$
BEGIN
    IF NOT pg_has_role(current_user, 'cuba_admin', 'MEMBER') THEN
        RAISE EXCEPTION
            'brain_audit_log is append-only (CFR-21). Tamper attempt by %', current_user;
    END IF;
    RETURN OLD;
END;
$func$ LANGUAGE plpgsql;
