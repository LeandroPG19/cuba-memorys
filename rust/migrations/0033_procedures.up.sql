
CREATE TABLE IF NOT EXISTS brain_procedures (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    name            TEXT NOT NULL,
    trigger_context TEXT NOT NULL DEFAULT '',

    steps           JSONB NOT NULL DEFAULT '[]'::jsonb,

    preconditions   TEXT NOT NULL DEFAULT '',
    verification    TEXT NOT NULL DEFAULT '',

    success_count   INTEGER NOT NULL DEFAULT 0,
    failure_count   INTEGER NOT NULL DEFAULT 0,
    last_outcome    TEXT CHECK (last_outcome IN ('success', 'failure')),
    last_used_at    TIMESTAMPTZ,

    project_id      UUID REFERENCES brain_projects(id) ON DELETE SET NULL,
    embedding       vector(384),
    embedding_model TEXT,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT uq_procedure_name_project UNIQUE (name, project_id)
);

CREATE INDEX IF NOT EXISTS idx_procedures_project ON brain_procedures (project_id);
CREATE INDEX IF NOT EXISTS idx_procedures_used ON brain_procedures (last_used_at DESC NULLS LAST);

COMMENT ON TABLE brain_procedures IS
    'Procedural memory (ACT-R): how something is done. Reinforced by outcome, not by access.';
