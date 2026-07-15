DROP TRIGGER IF EXISTS brain_audit_no_update ON brain_audit_log;
DROP TRIGGER IF EXISTS brain_audit_no_delete ON brain_audit_log;
DROP FUNCTION IF EXISTS brain_audit_block_mutation();
DROP TABLE IF EXISTS brain_audit_log CASCADE;
-- cuba_admin role kept (drop manually if no longer needed).
