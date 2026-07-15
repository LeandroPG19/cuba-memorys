#!/usr/bin/env bash

set -euo pipefail

if [[ $
  echo "usage: $0 <brain_*.dump>" >&2
  exit 1
fi

DUMP="$1"
DATABASE_URL="${DATABASE_URL:-postgresql://cuba:memorys2026@127.0.0.1:5488/brain}"

if [[ ! -f "$DUMP" ]]; then
  echo "error: dump file not found: $DUMP" >&2
  exit 1
fi

echo "Restoring $DUMP into $DATABASE_URL ..."
echo "This will DROP and recreate objects from the backup (--clean --if-exists)."
read -r -p "Continue? [y/N] " ans
if [[ "${ans,,}" != "y" ]]; then
  echo "Aborted."
  exit 0
fi

if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'cuba-memorys-db'; then
  echo "Using pg_restore inside container cuba-memorys-db ..."
  docker exec -i cuba-memorys-db pg_restore -U cuba -d brain \
    --clean \
    --if-exists \
    --no-owner \
    --no-acl \
    --verbose <"$DUMP"
elif command -v pg_restore >/dev/null 2>&1; then
  pg_restore \
    --dbname="$DATABASE_URL" \
    --clean \
    --if-exists \
    --no-owner \
    --no-acl \
    --verbose \
    "$DUMP"
else
  echo "error: pg_restore not found and cuba-memorys-db container not running." >&2
  exit 1
fi

echo "Restore finished."