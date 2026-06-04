#!/usr/bin/env bash
# Full logical backup of the cuba-memorys PostgreSQL database (schema + data).
# Uses pg_dump custom format (-Fc) for compression and pg_restore compatibility.
#
# Usage:
#   ./scripts/backup-db.sh
#   DATABASE_URL=postgresql://cuba:pass@host:5488/brain ./scripts/backup-db.sh
#   BACKUP_DIR=/path/to/backups ./scripts/backup-db.sh
#
# When the host pg_dump version does not match PG18 in Docker, the script uses
# `docker exec cuba-memorys-db` automatically if that container is running.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKUP_DIR="${BACKUP_DIR:-$ROOT/backups}"
KEEP_COUNT="${KEEP_COUNT:-7}"
DATABASE_URL="${DATABASE_URL:-postgresql://cuba:memorys2026@127.0.0.1:5488/brain}"

mkdir -p "$BACKUP_DIR"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$BACKUP_DIR/brain_${STAMP}.dump"
META="$BACKUP_DIR/brain_${STAMP}.meta.json"

dump_via_docker() {
  docker exec cuba-memorys-db pg_dump -U cuba -d brain \
    --format=custom \
    --no-owner \
    --no-acl \
    >"$OUT"
}

dump_via_host() {
  pg_dump "$DATABASE_URL" \
    --format=custom \
    --no-owner \
    --no-acl \
    --file="$OUT"
}

echo "Backing up database to $OUT ..."

if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'cuba-memorys-db'; then
  echo "Using pg_dump from container cuba-memorys-db (PG18)."
  dump_via_docker
elif command -v pg_dump >/dev/null 2>&1; then
  echo "Using host pg_dump."
  dump_via_host
else
  echo "error: no pg_dump and cuba-memorys-db container not running." >&2
  exit 1
fi

SIZE_BYTES="$(stat -c%s "$OUT" 2>/dev/null || stat -f%z "$OUT")"
TABLES=0
if command -v psql >/dev/null 2>&1; then
  TABLES="$(psql "$DATABASE_URL" -Atc \
    "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public' AND table_name LIKE 'brain_%'" 2>/dev/null || echo 0)"
elif docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'cuba-memorys-db'; then
  TABLES="$(docker exec cuba-memorys-db psql -U cuba -d brain -Atc \
    "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public' AND table_name LIKE 'brain_%'")"
fi

cat >"$META" <<EOF
{
  "created_at_utc": "$STAMP",
  "database": "brain",
  "dump_file": "$(basename "$OUT")",
  "size_bytes": $SIZE_BYTES,
  "brain_tables": $TABLES,
  "tool": "pg_dump",
  "format": "custom"
}
EOF

echo "Wrote metadata $META"
echo "Backup size: $SIZE_BYTES bytes ($(du -h "$OUT" | cut -f1))"

mapfile -t OLD < <(ls -1t "$BACKUP_DIR"/brain_*.dump 2>/dev/null || true)
if ((${#OLD[@]} > KEEP_COUNT)); then
  for f in "${OLD[@]:KEEP_COUNT}"; do
    base="${f%.dump}"
    rm -f "$f" "${base}.meta.json"
    echo "Pruned old backup: $(basename "$f")"
  done
fi

echo "Done. Latest backup: $OUT"