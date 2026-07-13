# Entorno del SANDBOX de cuba-memorys.
#   source sandbox/env.sh
#
# Aislamiento total respecto de producción:
#   - Contenedor Postgres propio (cuba-sandbox-db), volumen propio, puerto 5491.
#   - Binario compilado en el target/ de ESTE worktree, nunca el release/ vivo.
# Producción (INTOCABLE): contenedor cuba-memorys-db, volumen cuba_memorys_data,
# puerto 5488, DB `brain`, servida por 3 procesos MCP vivos.

SANDBOX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SANDBOX_ROOT

export CUBA_SANDBOX_PORT="${CUBA_SANDBOX_PORT:-5491}"
export CUBA_SANDBOX_CONTAINER="${CUBA_SANDBOX_CONTAINER:-cuba-sandbox-db}"
export CUBA_SANDBOX_VOLUME="${CUBA_SANDBOX_VOLUME:-cuba_sandbox_data}"

# Puerto de producción — prohibido. La guarda de abajo aborta si aparece.
CUBA_PROD_PORT=5488

# Password from the environment, with the docker-compose default as fallback.
# Not a secret (local container, bound to 127.0.0.1) but there is no reason to
# hardcode one more copy of it.
export CUBA_SANDBOX_PASSWORD="${CUBA_SANDBOX_PASSWORD:-memorys2026}"
export DATABASE_URL="postgresql://cuba:${CUBA_SANDBOX_PASSWORD}@127.0.0.1:${CUBA_SANDBOX_PORT}/brain"
export PGPASSWORD="$CUBA_SANDBOX_PASSWORD"

# Modelo de embeddings: e5-small 384-d (baseline, read-only, compartido).
# Para probar bge-m3 1024-d: source sandbox/env.bge-m3.sh
export ONNX_MODEL_PATH="${ONNX_MODEL_PATH:-/home/leandro/.cache/cuba-memorys/models}"
export ORT_DYLIB_PATH="${ORT_DYLIB_PATH:-/home/leandro/.cache/cuba-memorys/onnxruntime/libonnxruntime.so}"

# Binario del sandbox (target/ propio de este worktree).
export CUBA_SANDBOX_BIN="${SANDBOX_ROOT}/rust/target/release/cuba-memorys"

# ---------------------------------------------------------------------------
# GUARDA DURA: nada del sandbox puede apuntar a la base viva.
# ---------------------------------------------------------------------------
cuba_sandbox_guard() {
    local url="${DATABASE_URL:-}"
    if [[ "$url" == *":${CUBA_PROD_PORT}/"* ]]; then
        echo "ABORTADO: DATABASE_URL apunta al puerto de PRODUCCIÓN (${CUBA_PROD_PORT})." >&2
        echo "  URL: $url" >&2
        unset DATABASE_URL
        return 1
    fi
    if [[ "$CUBA_SANDBOX_BIN" == *"/cuba-memorys/rust/target"* ]]; then
        echo "ABORTADO: el binario apunta al checkout de PRODUCCIÓN (el que sirven los MCP vivos)." >&2
        return 1
    fi
    return 0
}

if cuba_sandbox_guard; then
    echo "[sandbox] OK  db=127.0.0.1:${CUBA_SANDBOX_PORT}/brain (contenedor ${CUBA_SANDBOX_CONTAINER})"
    echo "[sandbox] bin=${CUBA_SANDBOX_BIN}"
    echo "[sandbox] producción (5488 / brain / binario vivo) NO tocada"
else
    echo "[sandbox] entorno NO cargado — corregí lo de arriba antes de seguir." >&2
fi

alias pgsb='psql -h 127.0.0.1 -p ${CUBA_SANDBOX_PORT} -U cuba -d brain'
