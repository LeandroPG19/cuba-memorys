#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUST_DIR="$ROOT/rust"
export DATABASE_URL="${DATABASE_URL:-postgresql://cuba:memorys2026@127.0.0.1:5488/brain}"
export CUBA_JUDGE="${CUBA_JUDGE:-heuristic}"

cd "$RUST_DIR"

echo "=== cargo fmt --check ==="
cargo fmt --check

echo "=== cargo clippy (--all-targets: without it, tests/ is never linted) ==="
cargo clippy --all-targets -- -D warnings

echo "=== cargo test (unit + smoke) ==="
cargo test

echo "=== DB integration tests (--ignored) ==="
cargo test --test integration --test v08_project_scoping --test v09_integration \
           --test v021_audit_append_under_app_role -- --ignored --nocapture
cargo test --lib -- --ignored --nocapture

# build-gpu.sh, not a bare `cargo build --release`. Without --features cuda,
# gpu::wants_gpu() returns false unconditionally, CUBA_RERANK_DEVICE=gpu goes
# inert, and the reranker runs on CPU at 20.669s per query against 0.356s — 58x,
# measured in d7922fa. The E2E allows 15s per call, so every one of its 40 calls
# times out and the gate can never pass on a machine that has a GPU configured.
# It also overwrote the developer's GPU binary at this exact path.
echo "=== release build (same feature set production runs) ==="
"$ROOT/scripts/build-gpu.sh"

echo "=== E2E (25 MCP tools, subprocess per call) ==="
export CUBA_BINARY_PATH="$RUST_DIR/target/release/cuba-memorys"
python3 tests/e2e_all_tools.py

echo "=== MCP live session (single process, initialize + tools/list + calls) ==="
python3 "$ROOT/scripts/mcp_live_session_test.py"

echo "=== eval harness smoke (non-mutating; verifies it runs end-to-end) ==="
"$RUST_DIR/target/release/cuba-memorys" eval --dataset "$RUST_DIR/eval-datasets/smoke.jsonl" --k 10

echo ""
echo "All tests passed."