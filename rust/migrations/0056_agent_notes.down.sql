DROP TRIGGER IF EXISTS brain_agent_notes_cap ON brain_triggers;
DROP FUNCTION IF EXISTS brain_cap_agent_notes();
ALTER TABLE brain_triggers DROP COLUMN IF EXISTS from_agent;
