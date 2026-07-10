#!/usr/bin/env bash
# Run the full cuba-memorys test matrix (unit + DB integration + E2E).
#
# Requires:
#   - Rust toolchain
#   - DATABASE_URL pointing at pgvector Postgres (default :5488)
#   - Python 3 for E2E
#   - release binary built (or builds it)

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
cargo test --test integration --test v08_project_scoping --test v09_integration -- --ignored --nocapture

echo "=== release build ==="
cargo build --release

echo "=== E2E (25 MCP tools, subprocess per call) ==="
export CUBA_BINARY_PATH="$RUST_DIR/target/release/cuba-memorys"
python3 tests/e2e_all_tools.py

echo "=== MCP live session (single process, initialize + tools/list + calls) ==="
python3 "$ROOT/scripts/mcp_live_session_test.py"

echo ""
echo "All tests passed."