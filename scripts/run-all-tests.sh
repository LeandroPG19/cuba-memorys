#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUST_DIR="$ROOT/rust"
LIVE_DATABASE_URL="${DATABASE_URL:-postgresql://cuba:memorys2026@127.0.0.1:5488/brain}"
export CUBA_JUDGE="${CUBA_JUDGE:-heuristic}"

CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/cuba-memorys"
export ONNX_MODEL_PATH="${ONNX_MODEL_PATH:-$CACHE/models-bge-m3}"
export ORT_DYLIB_PATH="${ORT_DYLIB_PATH:-$CACHE/onnxruntime/libonnxruntime.so}"
export CUBA_RERANKER_PATH="${CUBA_RERANKER_PATH:-$CACHE/reranker-fused}"
export CUBA_NLI_PATH="${CUBA_NLI_PATH:-$CACHE/models-nli}"
export CUBA_EMBEDDING_DIM="${CUBA_EMBEDDING_DIM:-1024}"
export CUBA_EMBED_MODEL="${CUBA_EMBED_MODEL:-bge-m3}"
export CUBA_POOLING="${CUBA_POOLING:-cls}"

run_if_present() {
  local what="$1" probe="$2"
  shift 2
  if [[ -e "$probe" ]]; then
    echo "=== $what ==="
    "$@"
  else
    echo "SKIPPED: $what — nothing at $probe"
    echo "         These targets are NOT covered by this run. Install with"
    echo "         \`cuba-memorys models all\` to make them run."
  fi
}

GATE_DB="${GATE_DB:-brain_gate}"
GATE_DATABASE_URL="${LIVE_DATABASE_URL%/*}/$GATE_DB"

cd "$RUST_DIR"

provision_gate_db() {
  docker exec cuba-memorys-db psql -U cuba -d postgres -q \
    -c "DROP DATABASE IF EXISTS $GATE_DB WITH (FORCE)" \
    -c "CREATE DATABASE $GATE_DB" >/dev/null
  DATABASE_URL="$GATE_DATABASE_URL" CUBA_APP_ROLE=0 ONNX_MODEL_PATH="" \
    timeout 300 cargo run --quiet --bin cuba-memorys -- doctor >/dev/null 2>&1 || true
  if [[ "$CUBA_EMBEDDING_DIM" != "384" ]]; then
    DATABASE_URL="$GATE_DATABASE_URL" "$ROOT/scripts/migrate-embedding-dim.sh" \
      "$CUBA_EMBEDDING_DIM" >/dev/null 2>&1 || {
        echo "FAIL: could not retype the throwaway database to vector($CUBA_EMBEDDING_DIM)." >&2
        exit 1
      }
  fi
  local dim
  dim="$(docker exec cuba-memorys-db psql -U cuba -d "$GATE_DB" -Atc \
    "SELECT atttypmod FROM pg_attribute WHERE attrelid='brain_observations'::regclass AND attname='embedding'")"
  if [[ "$dim" != "$CUBA_EMBEDDING_DIM" ]]; then
    echo "FAIL: throwaway database is vector($dim) but the model produces $CUBA_EMBEDDING_DIM." >&2
    echo "      Every embedding write would fail with 'expected $dim dimensions'." >&2
    exit 1
  fi
  local tables
  tables="$(docker exec cuba-memorys-db psql -U cuba -d "$GATE_DB" -Atc \
    "SELECT count(*) FROM information_schema.tables WHERE table_schema='public'")"
  if ((tables < 20)); then
    echo "FAIL: could not migrate the throwaway database (only $tables tables)." >&2
    exit 1
  fi
  echo "OK  throwaway database $GATE_DB ready ($tables tables, vector($dim))"
}

drop_gate_db() {
  docker exec cuba-memorys-db psql -U cuba -d postgres -q \
    -c "DROP DATABASE IF EXISTS $GATE_DB WITH (FORCE)" >/dev/null 2>&1 || true
}
trap drop_gate_db EXIT

echo "=== cargo fmt --check ==="
cargo fmt --check

echo "=== cargo clippy (--all-targets: without it, tests/ is never linted) ==="
cargo clippy --all-targets -- -D warnings

echo "=== cargo test (unit + smoke) ==="
DATABASE_URL="$GATE_DATABASE_URL" cargo test

echo "=== throwaway database for every mutating step ==="
provision_gate_db

echo "=== DB integration tests (--ignored) ==="
export DATABASE_URL="$GATE_DATABASE_URL"
cargo test --test integration --test v08_project_scoping --test v09_integration \
           --test v021_audit_append_under_app_role --test v021_audit_downgrade \
           --test v021_import_quarantine_all_kinds \
           --test v022_merge_reports_what_it_drops --test v022_sync_takes_a_lock \
           --test v016_quarantine --test v016_relation_extraction \
           --test v017_codegraph_hardening \
           -- --ignored --nocapture
cargo test --lib -- --ignored --nocapture

echo "=== admin-role tests (they rewrite roles and the audit log) ==="
CUBA_APP_ROLE=0 cargo test --test v020_audit_update_applies --test v020_role_separation \
           -- --ignored --nocapture

run_if_present "tests that need the embedding model" "$ONNX_MODEL_PATH/model_quantized.onnx" \
  cargo test --test v016_chunking -- --ignored --nocapture

run_if_present "tests that need the NLI model" "$CUBA_NLI_PATH" \
  cargo test --test nli_entailment -- --ignored --nocapture

if command -v claude >/dev/null 2>&1; then
  echo "=== tests that need a local LLM CLI ==="
  cargo test --test v016_extract_without_sampling --test v017_relation_scan \
             -- --ignored --nocapture
else
  echo "SKIPPED: tests that need a local LLM CLI — no \`claude\` on PATH."
  echo "         v016_extract_without_sampling and v017_relation_scan are NOT covered."
fi

# build-gpu.sh, not a bare `cargo build --release`. Without --features cuda,
# gpu::wants_gpu() returns false unconditionally, CUBA_RERANK_DEVICE=gpu goes
# inert, and the reranker runs on CPU at 20.669s per query against 0.356s — 58x,
# measured in d7922fa. The E2E allows 15s per call, so every one of its 40 calls
# times out and the gate can never pass on a machine that has a GPU configured.
# It also overwrote the developer's GPU binary at this exact path.
echo "=== release build (same feature set production runs) ==="
"$ROOT/scripts/build-gpu.sh"

run_if_present "reranker tests (release: 387s in debug, seconds here)" \
  "$CUBA_RERANKER_PATH/model.onnx" \
  cargo test --release --test v017_rerank_gpu -- --ignored --nocapture

echo "=== E2E (25 MCP tools, subprocess per call) ==="
export CUBA_BINARY_PATH="$RUST_DIR/target/release/cuba-memorys"
python3 tests/e2e_all_tools.py

echo "=== MCP live session (single process, initialize + tools/list + calls) ==="
python3 "$ROOT/scripts/mcp_live_session_test.py"

echo "=== eval harness smoke (read-only, so it runs against the real corpus) ==="
DATABASE_URL="$LIVE_DATABASE_URL" \
  "$RUST_DIR/target/release/cuba-memorys" eval \
  --dataset "$RUST_DIR/eval-datasets/smoke.jsonl" --k 10

echo ""
echo "All tests passed."
