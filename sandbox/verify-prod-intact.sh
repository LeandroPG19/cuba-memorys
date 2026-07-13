#!/usr/bin/env bash
# Comprueba que la base VIVA (brain, puerto 5488) sigue intacta.
# Se corre después de cada prueba del sandbox. Si algún conteo baja, algo tocó producción.
#
# La línea base se guarda en sandbox/.prod-baseline la primera vez.
set -uo pipefail

PROD_PORT=5488
BASELINE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.prod-baseline"

read -r -d '' Q <<'SQL' || true
SELECT (SELECT count(*) FROM brain_observations)||'/'
    || (SELECT count(*) FROM brain_entities)||'/'
    || (SELECT count(*) FROM brain_facts);
SQL

now="$(PGPASSWORD=memorys2026 psql -h 127.0.0.1 -p "$PROD_PORT" -U cuba -d brain -tAc "$Q" 2>/dev/null | tr -d '[:space:]')"

if [[ -z "$now" ]]; then
    echo "[prod] no se pudo leer brain en :$PROD_PORT (¿contenedor parado?) — sin verificar"
    exit 0
fi

if [[ ! -f "$BASELINE" ]]; then
    echo "$now" > "$BASELINE"
    echo "[prod] línea base registrada: obs/ent/facts = $now"
    exit 0
fi

base="$(tr -d '[:space:]' < "$BASELINE")"
if [[ "$now" == "$base" ]]; then
    echo "[prod] INTACTA — obs/ent/facts = $now (== línea base)"
else
    # Crecer es normal: los MCP vivos escriben durante la sesión. Encoger NO lo es.
    IFS=/ read -r o1 e1 f1 <<< "$base"
    IFS=/ read -r o2 e2 f2 <<< "$now"
    if (( o2 < o1 || e2 < e1 || f2 < f1 )); then
        echo "[prod] ALERTA: algún conteo BAJÓ  base=$base  ahora=$now" >&2
        exit 1
    fi
    echo "[prod] intacta (creció por uso normal de los MCP vivos): base=$base ahora=$now"
    echo "$now" > "$BASELINE"
fi
