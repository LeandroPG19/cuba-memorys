#!/usr/bin/env bash
# Levanta el sandbox desde cero: contenedor Postgres aislado + restauración del
# backup known-good. Idempotente: si el contenedor ya existe y está sano, no hace nada.
#
#   ./sandbox/up.sh            # levanta (reusa si ya está)
#   ./sandbox/up.sh --fresh    # destruye y recrea desde el backup
set -euo pipefail

SANDBOX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${CUBA_SANDBOX_PORT:-5491}"
CONTAINER="${CUBA_SANDBOX_CONTAINER:-cuba-sandbox-db}"
VOLUME="${CUBA_SANDBOX_VOLUME:-cuba_sandbox_data}"
IMAGE="pgvector/pgvector:pg18"   # idéntica a producción
PROD_PORT=5488

# Backup de referencia: el known-good verificado (dos formatos + COMO-RESTAURAR.md).
BACKUP="${CUBA_SANDBOX_BACKUP:-/home/leandro/proyectos/MCP/cuba-memorys/backups/verified/brain_known_good_20260710T163438Z.dump}"

if [[ "$PORT" == "$PROD_PORT" ]]; then
    echo "ABORTADO: el sandbox no puede usar el puerto de producción ($PROD_PORT)." >&2
    exit 1
fi

if [[ "${1:-}" == "--fresh" ]]; then
    echo "[sandbox] --fresh: destruyendo contenedor y volumen del sandbox"
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    docker volume rm "$VOLUME" >/dev/null 2>&1 || true
fi

if ! docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "[sandbox] creando contenedor $CONTAINER en 127.0.0.1:$PORT"
    docker volume create "$VOLUME" >/dev/null
    docker run -d --name "$CONTAINER" \
        -e POSTGRES_USER=cuba -e POSTGRES_PASSWORD=memorys2026 -e POSTGRES_DB=brain \
        -p "127.0.0.1:${PORT}:5432" \
        -v "${VOLUME}:/var/lib/postgresql" \
        --health-cmd="pg_isready -U cuba -d brain" \
        --health-interval=3s --health-timeout=3s --health-retries=20 \
        "$IMAGE" >/dev/null
    NEEDS_RESTORE=1
else
    docker start "$CONTAINER" >/dev/null 2>&1 || true
    NEEDS_RESTORE=0
fi

echo -n "[sandbox] esperando healthy"
for _ in $(seq 1 40); do
    st=$(docker inspect "$CONTAINER" --format '{{.State.Health.Status}}' 2>/dev/null || echo "?")
    [[ "$st" == "healthy" ]] && { echo " → ok"; break; }
    echo -n "."; sleep 2
done

if [[ "$NEEDS_RESTORE" == "1" ]]; then
    "${SANDBOX_ROOT}/sandbox/reset.sh"
fi

docker ps --filter "name=$CONTAINER" --format '[sandbox] {{.Names}}  {{.Status}}  {{.Ports}}'
"${SANDBOX_ROOT}/sandbox/verify-prod-intact.sh"
