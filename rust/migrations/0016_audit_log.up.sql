-- V0.9: Tamper-evident audit log with hash chain (CFR-21 Part 11 inspired).
--
-- Each row's `current_hash` = sha256(prev_hash || action || payload_canonical
-- || created_at_iso8601). Verifying the chain means walking the table by id
-- and recomputing each hash; any tampering breaks the chain at that row.
--
-- Append-only enforced via PostgreSQL trigger: UPDATE/DELETE raise EXCEPTION
-- (only role `cuba_admin` can bypass — for legal data deletion under GDPR).
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'brain_audit_log'
    ) THEN
        CREATE TABLE brain_audit_log (
            id BIGSERIAL PRIMARY KEY,
            prev_hash BYTEA,
            action TEXT NOT NULL,
            payload JSONB NOT NULL DEFAULT '{}',
            current_hash BYTEA NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX idx_audit_action ON brain_audit_log(action);
        CREATE INDEX idx_audit_created ON brain_audit_log(created_at DESC);

        -- Trigger: refuse UPDATE / DELETE from any role other than cuba_admin.
        -- Keeps the chain immutable in normal operation.
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

        CREATE TRIGGER brain_audit_no_update
            BEFORE UPDATE ON brain_audit_log
            FOR EACH ROW EXECUTE FUNCTION brain_audit_block_mutation();
        CREATE TRIGGER brain_audit_no_delete
            BEFORE DELETE ON brain_audit_log
            FOR EACH ROW EXECUTE FUNCTION brain_audit_block_mutation();

        -- Optional admin role for emergency rectification (e.g. GDPR cascade).
        -- Created without password — must be explicitly GRANTed in production.
        DO $role$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'cuba_admin') THEN
                CREATE ROLE cuba_admin NOLOGIN;
            END IF;
        END
        $role$;
    END IF;
END $$;
