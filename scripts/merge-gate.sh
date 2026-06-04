#!/usr/bin/env bash
# Pre-merge gate — run this once before merging feature/mejoras → main.
# Exits non-zero on any failure.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  CUBA-MEMORYS MERGE GATE                                 ║"
echo "╚══════════════════════════════════════════════════════════╝"

# 1. DB must be reachable
export DATABASE_URL="${DATABASE_URL:-postgresql://cuba:memorys2026@127.0.0.1:5488/brain}"
if command -v pg_isready >/dev/null 2>&1; then
  pg_isready -h 127.0.0.1 -p 5488 -U cuba -d brain >/dev/null \
    || { echo "FAIL: Postgres not ready on :5488"; exit 1; }
  echo "OK  Postgres :5488"
else
  docker exec cuba-memorys-db pg_isready -U cuba -d brain >/dev/null \
    || { echo "FAIL: cuba-memorys-db container not healthy"; exit 1; }
  echo "OK  Postgres (docker)"
fi

# 2. Backup before merge-affecting verification (optional but recommended)
if [[ "${SKIP_BACKUP:-0}" != "1" ]]; then
  "$ROOT/scripts/backup-db.sh"
  echo "OK  Database backup"
fi

# 3. Full test matrix
"$ROOT/scripts/run-all-tests.sh"

# 4. cargo audit (matches CI)
echo "=== cargo audit ==="
(cd "$ROOT/rust" && cargo audit)

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  MERGE GATE PASSED — safe to merge                       ║"
echo "╚══════════════════════════════════════════════════════════╝"