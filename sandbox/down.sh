#!/usr/bin/env bash
# Para el sandbox (conserva el volumen; usar up.sh --fresh para empezar de cero).
set -euo pipefail
CONTAINER="${CUBA_SANDBOX_CONTAINER:-cuba-sandbox-db}"
docker stop "$CONTAINER" >/dev/null 2>&1 && echo "[sandbox] $CONTAINER detenido" || echo "[sandbox] $CONTAINER ya estaba parado"
echo "[sandbox] producción intacta — nunca se tocó el contenedor cuba-memorys-db"
