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

# Every vector column, discovered — not a hand-maintained list.
#
# This script used to name brain_observations and brain_episodes explicitly, and
# that is exactly how it broke: migration 0033 added brain_procedures with its own
# vector column, the script did not know about it, and the table stayed at 384-d
# while the rest of the corpus moved to 1024. Every INSERT into it then failed
# with a dimension error that surfaced as an opaque "guardando el procedimiento".
#
# A table added tomorrow would have been forgotten the same way. So: ask the
# catalog which columns are vectors, and migrate all of them.
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 <<SQL
BEGIN;

DO \$\$
DECLARE
    r RECORD;
    idx RECORD;
BEGIN
    -- HNSW indexes are dimension-bound: they must go before the retype and be
    -- rebuilt after. Dropped by discovery too, for the same reason.
    FOR idx IN
        SELECT i.indexname
        FROM pg_indexes i
        WHERE i.schemaname = 'public' AND i.indexdef ILIKE '%USING hnsw%'
    LOOP
        EXECUTE format('DROP INDEX IF EXISTS %I', idx.indexname);
        RAISE NOTICE '[migrate] dropped index %', idx.indexname;
    END LOOP;

    FOR r IN
        SELECT c.relname AS tbl, a.attname AS col,
               EXISTS (
                   SELECT 1 FROM pg_attribute m
                   WHERE m.attrelid = c.oid AND m.attname = 'embedding_model'
                     AND m.attnum > 0 AND NOT m.attisdropped
               ) AS has_model
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
          AND c.relkind = 'r'
          AND a.attnum > 0 AND NOT a.attisdropped
          AND format_type(a.atttypid, a.atttypmod) LIKE 'vector(%'
    LOOP
        -- Vectors of the old dimension cannot be cast, only dropped. Text is untouched.
        EXECUTE format('UPDATE %I SET %I = NULL', r.tbl, r.col);
        IF r.has_model THEN
            -- Blank the model tag so `reembed` knows these rows are stale.
            EXECUTE format('UPDATE %I SET embedding_model = NULL', r.tbl);
        END IF;
        EXECUTE format('ALTER TABLE %I ALTER COLUMN %I TYPE vector($DIM)', r.tbl, r.col);
        RAISE NOTICE '[migrate] % .% -> vector($DIM)', r.tbl, r.col;
    END LOOP;
END \$\$;

-- Rebuild the HNSW indexes on the two tables large enough to need them. A vector
-- index on a table with a dozen rows costs more than the sequential scan it saves.
CREATE INDEX idx_obs_embedding_hnsw ON brain_observations
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 128);
CREATE INDEX idx_episodes_embedding_hnsw ON brain_episodes
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 128);

COMMIT;
SQL

echo "[migrate] done. Now run: cuba-memorys reembed"
echo "[migrate] every vector column:"
psql "$DATABASE_URL" -tAc \
  "SELECT '  '||c.relname||'.'||a.attname||' -> '||format_type(a.atttypid, a.atttypmod)
   FROM pg_attribute a
   JOIN pg_class c ON c.oid = a.attrelid
   JOIN pg_namespace n ON n.oid = c.relnamespace
   WHERE n.nspname='public' AND c.relkind='r'
     AND a.attnum > 0 AND NOT a.attisdropped
     AND format_type(a.atttypid, a.atttypmod) LIKE 'vector(%'
   ORDER BY c.relname;"
