#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  CUBA-MEMORYS MERGE GATE                                 ║"
echo "╚══════════════════════════════════════════════════════════╝"

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

if [[ "${SKIP_BACKUP:-0}" != "1" ]]; then
  "$ROOT/scripts/backup-db.sh"
  echo "OK  Database backup"
fi

"$ROOT/scripts/run-all-tests.sh"

echo "=== cargo audit ==="
(cd "$ROOT/rust" && cargo audit)

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  MERGE GATE PASSED — safe to merge                       ║"
echo "╚══════════════════════════════════════════════════════════╝"