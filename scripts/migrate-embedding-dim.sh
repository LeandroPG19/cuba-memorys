#!/usr/bin/env bash
# Phase 5: migrate the pgvector embedding columns to a new dimension (e.g. 384
# e5-small -> 1024 bge-m3). Vector CONTENT is dropped (set NULL) because you
# cannot cast between dimensions; text/entities/facts are untouched. Re-embed
# afterwards with `cuba_zafra reembed`.
#
# DESTRUCTIVE to embeddings only. Take a backup first (scripts/backup-db.sh).
# Requires a role with DDL (superuser/owner) — pass it via DATABASE_URL.
#
# Usage:
#   DATABASE_URL="postgresql://cuba:...@127.0.0.1:5488/brain_dev" \
#     scripts/migrate-embedding-dim.sh 1024
set -euo pipefail

DIM="${1:?usage: migrate-embedding-dim.sh <DIM>}"
: "${DATABASE_URL:?set DATABASE_URL (needs DDL privileges)}"

echo "[migrate] target dimension: vector($DIM)"
echo "[migrate] this drops all embedding vectors (content preserved). Ctrl-C to abort."

psql "$DATABASE_URL" -v ON_ERROR_STOP=1 <<SQL
BEGIN;

-- Drop HNSW indexes (they are dimension-bound).
DROP INDEX IF EXISTS idx_obs_embedding_hnsw;
DROP INDEX IF EXISTS idx_episodes_embedding_hnsw;

-- brain_observations: clear vectors, retype, mark model stale so reembed picks them.
UPDATE brain_observations SET embedding = NULL, embedding_model = NULL;
ALTER TABLE brain_observations ALTER COLUMN embedding TYPE vector($DIM);

-- brain_episodes: same.
UPDATE brain_episodes SET embedding = NULL;
ALTER TABLE brain_episodes ALTER COLUMN embedding TYPE vector($DIM);

-- Recreate HNSW indexes with the original build params.
CREATE INDEX idx_obs_embedding_hnsw ON brain_observations
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 128);
CREATE INDEX idx_episodes_embedding_hnsw ON brain_episodes
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 128);

COMMIT;
SQL

echo "[migrate] done. Column type is now vector($DIM). Run cuba_zafra reembed to repopulate."
psql "$DATABASE_URL" -tAc \
  "SELECT 'obs col dim = '||atttypmod-4 FROM pg_attribute
   WHERE attrelid='brain_observations'::regclass AND attname='embedding';"
