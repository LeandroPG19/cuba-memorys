-- Quarantine arrived in 0037 for brain_observations only. But cuba_sync import
-- writes seven tables, and two of the others are served straight back to the
-- caller: faro reads brain_episodes, expediente reads brain_errors. A bundle is a
-- JSON file in a git repository that anyone with commit access can edit, so an
-- imported episode or error carrying a credential was stored trusted and served
-- with nothing able to hold it back. Same column, same two states, same meaning as
-- 0037 — the point is that every table an importer can write now shares one notion
-- of trust instead of one table having it and the rest being taken on faith.
--
-- Existing rows default to trusted, so nothing already stored changes visibility.

ALTER TABLE brain_episodes
    ADD COLUMN IF NOT EXISTS trust text NOT NULL DEFAULT 'trusted';

ALTER TABLE brain_episodes
    ADD CONSTRAINT brain_episodes_trust_check
    CHECK (trust IN ('trusted', 'quarantined'));

ALTER TABLE brain_errors
    ADD COLUMN IF NOT EXISTS trust text NOT NULL DEFAULT 'trusted';

ALTER TABLE brain_errors
    ADD CONSTRAINT brain_errors_trust_check
    CHECK (trust IN ('trusted', 'quarantined'));

CREATE INDEX IF NOT EXISTS idx_episodes_quarantined
    ON brain_episodes (created_at DESC)
    WHERE trust = 'quarantined';

CREATE INDEX IF NOT EXISTS idx_errors_quarantined
    ON brain_errors (created_at DESC)
    WHERE trust = 'quarantined';

COMMENT ON COLUMN brain_episodes.trust IS
    'trusted = retrievable; quarantined = imported from untrusted text, withheld from cuba_faro until promoted via cuba_eco action=promote kind=episode.';

COMMENT ON COLUMN brain_errors.trust IS
    'trusted = retrievable; quarantined = imported from untrusted text, withheld from cuba_expediente until promoted via cuba_eco action=promote kind=error.';
