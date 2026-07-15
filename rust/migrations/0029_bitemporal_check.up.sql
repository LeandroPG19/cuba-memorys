
ALTER TABLE brain_facts
    ADD CONSTRAINT ck_facts_valid_interval
    CHECK (valid_to IS NULL OR valid_from <= valid_to);

ALTER TABLE brain_facts
    ADD CONSTRAINT ck_facts_current_open
    CHECK (NOT is_current OR valid_to IS NULL);
