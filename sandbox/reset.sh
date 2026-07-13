#!/usr/bin/env bash
# Devuelve la base del sandbox al estado known-good verificado.
# Útil entre pruebas destructivas: cada mejora se prueba sobre una base limpia.
set -euo pipefail

PORT="${CUBA_SANDBOX_PORT:-5491}"
CONTAINER="${CUBA_SANDBOX_CONTAINER:-cuba-sandbox-db}"
BACKUP="${CUBA_SANDBOX_BACKUP:-/home/leandro/proyectos/MCP/cuba-memorys/backups/verified/brain_known_good_20260710T163438Z.dump}"

[[ -f "$BACKUP" ]] || { echo "ABORTADO: no existe el backup $BACKUP" >&2; exit 1; }
[[ "$PORT" == "5488" ]] && { echo "ABORTADO: puerto de producción." >&2; exit 1; }

echo "[sandbox] restaurando brain desde $(basename "$BACKUP")"
docker exec "$CONTAINER" psql -U cuba -d postgres -q -c "DROP DATABASE IF EXISTS brain WITH (FORCE);"
docker exec "$CONTAINER" psql -U cuba -d postgres -q -c "CREATE DATABASE brain OWNER cuba;"
docker cp "$BACKUP" "${CONTAINER}:/tmp/brain.dump" >/dev/null
docker exec "$CONTAINER" pg_restore -U cuba -d brain --no-owner --no-privileges /tmp/brain.dump 2>/dev/null || true

docker exec "$CONTAINER" psql -U cuba -d brain -tAc \
    "SELECT '[sandbox] restaurado: obs='||(SELECT count(*) FROM brain_observations)
     ||' ent='||(SELECT count(*) FROM brain_entities)
     ||' facts='||(SELECT count(*) FROM brain_facts)
     ||' tablas='||(SELECT count(*) FROM information_schema.tables
                    WHERE table_schema='public' AND table_name LIKE 'brain%');"
