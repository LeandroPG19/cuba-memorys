-- 0033: procedural memory — the fourth memory.
--
-- The taxonomy the field converged on has four stores: episodic (what happened),
-- semantic (what is true), PROCEDURAL (how something is done), and working. cuba
-- had three. Its eight observation types — fact, decision, lesson, preference,
-- context, tool_usage, error, solution — can all record that something IS the
-- case. None of them records how to DO something: how this project is brought up,
-- what the test command is, which order the migrations and the reembed go in.
--
-- That is precisely the knowledge an agent rediscovers every session, burning
-- context and getting it wrong on the way.
--
-- ## Why this is a table and not a ninth observation type
--
-- ACT-R (Anderson & Lebiere, CMU) separates declarative memory (chunks) from
-- procedural memory (production rules), and the two are reinforced by different
-- signals. Declarative memory strengthens with ACCESS — cuba already models this
-- with Hebbian boosts and the testing effect on `access_count`. Procedural memory
-- strengthens with SUCCESS.
--
-- The distinction is not academic. If a procedure were stored as an observation,
-- it would gain importance every time it was retrieved — so a recipe that is
-- consulted constantly BECAUSE IT KEEPS FAILING would climb the rankings. Access
-- is not evidence of value. Outcome is. A procedure needs a store whose
-- reinforcement signal is whether it worked.

CREATE TABLE IF NOT EXISTS brain_procedures (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- What it is called, and when it applies. `trigger_context` is the IF half of
    -- an ACT-R production rule: the condition under which this procedure is the
    -- right one to run ("when the dev services need to come up on mapupita-web").
    name            TEXT NOT NULL,
    trigger_context TEXT NOT NULL DEFAULT '',

    -- The THEN half. JSONB rather than prose so it can be rendered to markdown,
    -- to a SKILL.md, or eventually executed — without re-parsing free text.
    -- Each step: {"do": "...", "run": "cmd" | null, "expect": "..." | null}
    steps           JSONB NOT NULL DEFAULT '[]'::jsonb,

    -- What must already be true, and how you know it worked. The verification is
    -- the part people leave out, and it is the part that makes a procedure
    -- trustworthy instead of hopeful.
    preconditions   TEXT NOT NULL DEFAULT '',
    verification    TEXT NOT NULL DEFAULT '',

    -- Utility, learned from outcomes. NOT from access.
    success_count   INTEGER NOT NULL DEFAULT 0,
    failure_count   INTEGER NOT NULL DEFAULT 0,
    last_outcome    TEXT CHECK (last_outcome IN ('success', 'failure')),
    last_used_at    TIMESTAMPTZ,

    project_id      UUID REFERENCES brain_projects(id) ON DELETE SET NULL,
    -- Semantic search over name + trigger, same pipeline as observations.
    embedding       vector(384),
    embedding_model TEXT,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- One procedure per name per project. A second "levantar los servicios" in
    -- the same project is an edit, not a new recipe.
    CONSTRAINT uq_procedure_name_project UNIQUE (name, project_id)
);

CREATE INDEX IF NOT EXISTS idx_procedures_project ON brain_procedures (project_id);
CREATE INDEX IF NOT EXISTS idx_procedures_used ON brain_procedures (last_used_at DESC NULLS LAST);

-- Vector index only pays for itself past a few thousand rows; procedures are
-- inherently few (a project has a dozen, not a thousand), so a sequential scan
-- over them is faster than an ivfflat probe. Deliberately omitted.

COMMENT ON TABLE brain_procedures IS
    'Procedural memory (ACT-R): how something is done. Reinforced by outcome, not by access.';
