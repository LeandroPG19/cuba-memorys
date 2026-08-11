#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  CUBA-MEMORYS MERGE GATE                                 ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "WHAT THIS GATE DOES NOT CHECK — read before trusting a green run:"
echo "  · --features docs      CI compiles and tests it; this gate does not."
echo "  · the reranker         E2E forces CUBA_RERANKER_PATH at an empty dir,"
echo "                         so its 40 calls exercise the identity fallback."
echo "                         mcp_live_session_test.py inherits your shell and"
echo "                         DOES use it. Two suites, opposite policies."
echo "  · GPU placement        build-gpu.sh compiles --features cuda, but no"
echo "                         assertion here fails if the work lands on CPU."
echo "  · retrieval quality    the eval step is a smoke run. It proves the"
echo "                         harness executes; it asserts no nDCG threshold."
echo "  · other platforms      Linux x64 only. The published musl binary is"
echo "                         static-pie and cannot dlopen ONNX Runtime."
echo "  · migrations           the throwaway database is created from scratch, so"
echo "                         a migration that only works on an EXISTING schema"
echo "                         is still not covered here."
echo ""
echo "WHERE IT WRITES:"
echo "  · every mutating step  a throwaway database (GATE_DB, default brain_gate),"
echo "                         created before and dropped after. Your real corpus"
echo "                         is never a test fixture."
echo "  · the eval step        reads the REAL database, because a smoke run against"
echo "                         an empty corpus would prove nothing. Read-only."
echo ""

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