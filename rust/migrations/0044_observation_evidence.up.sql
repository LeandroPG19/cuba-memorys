-- `source` says who produced an observation and `trust` says whether it is
-- quarantined, but nothing said how strongly it is held. Everything in the graph
-- was equally believable: a model's guess and a fact parsed out of a real syntax
-- tree came back from the same search looking the same.
--
-- Three levels, and the only thing that makes them mean anything is who is allowed
-- to write which:
--
--   asserted  someone said so. The default, and the ONLY value the MCP path can
--             produce — no handler binds this column, so the default is not a
--             convention the handlers agree to follow, it is the only thing they
--             are able to do.
--   observed  derived mechanically from something real. codegraph writes this
--             because tree-sitter parsed it out of an actual AST.
--   verified  a recorded check was re-run and reproduced. CLI only.
--
-- `verification` is the check itself, in the same spirit as
-- brain_procedures.verification (0033), which has existed since v0.16 holding
-- "what must be true and how you know it worked" and has never been executed.
-- Existing rows default to asserted, which is what they always were.

ALTER TABLE brain_observations
    ADD COLUMN IF NOT EXISTS evidence text NOT NULL DEFAULT 'asserted';

ALTER TABLE brain_observations
    ADD CONSTRAINT brain_observations_evidence_check
    CHECK (evidence IN ('asserted', 'observed', 'verified'));

ALTER TABLE brain_observations
    ADD COLUMN IF NOT EXISTS verification text;

ALTER TABLE brain_observations
    ADD COLUMN IF NOT EXISTS verified_at timestamptz;

-- Anything above asserted has to say how it got there. A row claiming to be
-- verified with no record of what was re-run is exactly the unfalsifiable claim
-- these levels exist to prevent.
ALTER TABLE brain_observations
    ADD CONSTRAINT brain_observations_evidence_needs_a_check
    CHECK (evidence = 'asserted' OR verification IS NOT NULL);

ALTER TABLE brain_observations
    ADD CONSTRAINT brain_observations_verified_is_stamped
    CHECK ((evidence = 'verified') = (verified_at IS NOT NULL));

CREATE INDEX IF NOT EXISTS idx_obs_evidence
    ON brain_observations (evidence, verified_at DESC)
    WHERE evidence <> 'asserted';

COMMENT ON COLUMN brain_observations.evidence IS
    'asserted = someone said so (default, and all the MCP path can write); observed = derived mechanically, e.g. codegraph from a tree-sitter AST; verified = a recorded check was re-run and reproduced, written only by the CLI.';

COMMENT ON COLUMN brain_observations.verification IS
    'The check that supports this observation above asserted level: what was run, whose output has to be reproduced. Required whenever evidence is not asserted.';
