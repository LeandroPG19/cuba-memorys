#!/usr/bin/env bash
# Terminal demo for the README GIF — cuba-memorys v0.11
#
# Runs against a THROWAWAY Postgres it starts and destroys itself.
#
# The previous demo defaulted DATABASE_URL to the real brain on :5488, so every
# run wrote entities into someone's actual memory and ran PageRank over it. A demo
# you have to think twice about is a demo nobody runs. This one you can execute on
# any machine with Docker, and it leaves nothing behind.
#
#   ./scripts/demo.sh                      # ephemeral DB, cleaned up on exit
#   DEMO_DATABASE_URL=postgres://…  ./scripts/demo.sh   # bring your own
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINARY="${CUBA_BINARY_PATH:-$ROOT/rust/target/release/cuba-memorys}"

BLUE='\033[1;34m'; GREEN='\033[1;32m'; YELLOW='\033[1;33m'
CYAN='\033[1;36m'; MAGENTA='\033[1;35m'; RED='\033[1;31m'
DIM='\033[2m'; BOLD='\033[1m'; NC='\033[0m'

PAUSE="${DEMO_PAUSE:-0.3}"
[[ "${RECORD_DEMO:-}" == "1" ]] && PAUSE="1.6"

[[ -x "$BINARY" ]] || { echo "Binary not found: $BINARY (cd rust && cargo build --release)"; exit 1; }

# ── Hermetic: the demo does not inherit your embedding setup ─────────────────
#
# It did, once, and it blew up in a way worth keeping as a comment. A machine
# configured for bge-m3 (1024-d) exported CUBA_EMBEDDING_DIM=1024, the demo's fresh
# database came up with the default vector(384) column, and the server refused to
# start — correctly, loudly, and fatally for the demo. A demo that only runs on the
# author's machine is not a demo.
#
# So it pins its own: the default 384-d model, matching the schema a fresh database
# gets. Uses the real ONNX model when it is already downloaded; falls back to hash
# embeddings otherwise (the lexical and BM25 branches carry the search either way).
unset CUBA_EMBEDDING_DIM CUBA_EMBED_MODEL CUBA_POOLING \
      CUBA_QUERY_PREFIX CUBA_PASSAGE_PREFIX CUBA_TOOL_PROFILE CUBA_COMPACT_CHARS

# The model needs BOTH its weights and the ONNX Runtime library. With only the
# first, the server used to hang on every embedding without logging a word (fixed
# in v0.11.2 — it now says so and degrades). Take both or neither.
MODEL_DIR="${HOME}/.cache/cuba-memorys/models"
ORT_LIB="${HOME}/.cache/cuba-memorys/onnxruntime/libonnxruntime.so"
if [[ -d "$MODEL_DIR" && -f "$ORT_LIB" ]]; then
  export ONNX_MODEL_PATH="$MODEL_DIR" ORT_DYLIB_PATH="$ORT_LIB"
else
  unset ONNX_MODEL_PATH ORT_DYLIB_PATH   # hash embeddings; lexical + BM25 still work
fi

# `auto` and not `heuristic`: verification escalates evidence to a judge, and the
# heuristic one has no model to reason with — it answers "unknown" to everything,
# which is honest but shows nothing. With `auto` the demo uses whatever is there: an
# MCP client's own model, a local `claude` CLI, or — failing both — the heuristic,
# and then it says so rather than pretending.
export CUBA_JUDGE="${CUBA_JUDGE:-auto}"
if ! command -v claude >/dev/null 2>&1; then
  echo -e "${DIM}(no LLM judge on PATH — verification will answer 'unknown', which is the honest\n answer without one. Install the claude CLI to see it reach a verdict.)${NC}"
fi

CONTAINER="cuba-demo-$$"
PORT="${DEMO_PORT:-5499}"
WORKDIR="$(mktemp -d)"

cleanup() {
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  rm -rf "$WORKDIR"
}

# ── Banner first ─────────────────────────────────────────────────────────────
# Before the database, deliberately: the first frame of the README GIF is what
# people see before they press play, and "starting a throwaway postgres…" is not it.
clear
echo -e "${BOLD}${GREEN}"
cat <<'BANNER'
  ╔═══════════════════════════════════════════════════════════════╗
  ║  cuba-memorys — Persistent Memory for AI Agents               ║
  ║  28 MCP tools · 14 CLI commands · four kinds of memory        ║
  ╚═══════════════════════════════════════════════════════════════╝
BANNER
echo -e "${NC}"
echo -e "${YELLOW}# PyPI:${NC}  pip install cuba-memorys"
echo -e "${YELLOW}# npm:${NC}   npm install -g cuba-memorys"
echo -e "${MAGENTA}# MCP:${NC}   claude mcp add cuba-memorys -- cuba-memorys"
sleep 2

# ── Throwaway database ───────────────────────────────────────────────────────
if [[ -n "${DEMO_DATABASE_URL:-}" ]]; then
  export DATABASE_URL="$DEMO_DATABASE_URL"
  trap 'rm -rf "$WORKDIR"' EXIT INT TERM
else
  trap cleanup EXIT INT TERM
  echo -e "\n${DIM}(this demo runs on a throwaway postgres it removes on exit — it will not touch yours)${NC}"
  docker run -d --name "$CONTAINER" -p "${PORT}:5432" \
    -e POSTGRES_USER=demo -e POSTGRES_PASSWORD=demo -e POSTGRES_DB=brain \
    pgvector/pgvector:pg18 >/dev/null
  export DATABASE_URL="postgresql://demo:demo@127.0.0.1:${PORT}/brain"
  until docker exec "$CONTAINER" pg_isready -U demo -q 2>/dev/null; do sleep 0.5; done
  sleep 1
fi

# Whatever the demo writes to disk (the dashboard) lands here, not in your repo.
cd "$WORKDIR"

# One MCP round-trip. Prints the tool call, then what came back.
call() {
  local tool="$1" args="$2" label="$3"
  echo -e "\n${CYAN}>>> ${label}${NC}"
  echo -e "${DIM}cuba_${tool}(${args})${NC}"
  printf '%s\n' \
    '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"demo","version":"1"}}}' \
    "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"cuba_${tool}\",\"arguments\":${args}}}" \
  | timeout 120 "$BINARY" 2>/dev/null | tail -1 | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    r = d.get('result', d.get('error', {}))
    t = json.loads(r['content'][0]['text']) if 'content' in r else r
    print(json.dumps(t, indent=2, ensure_ascii=False)[:480])
except Exception:
    print('(no output)')
" 2>/dev/null || echo "(timeout)"
  sleep "$PAUSE"
}

# A CLI subcommand — the brain without an LLM in the middle.
cli() {
  local label="$1"; shift
  echo -e "\n${MAGENTA}\$ cuba-memorys $*${NC}"
  echo -e "${DIM}${label}${NC}"
  timeout 60 "$BINARY" "$@" 2>/dev/null | head -14
  sleep "$PAUSE"
}

# ── 1. Semantic + episodic: what the agent learns ────────────────────────────
echo -e "\n${BOLD}── The agent learns, and it sticks ──${NC}"
call alma '{"action":"create","name":"cuba-memorys","entity_type":"project"}' \
  "An entity in the knowledge graph"
call cronica '{"action":"add","entity_name":"cuba-memorys","content":"Hybrid retrieval fuses text, BM25 and pgvector with RRF (k=60). Ties break by id, so the benchmark is reproducible.","observation_type":"fact","source":"agent"}' \
  "An observation — deduped, importance-primed, embedded"
call cronica '{"action":"add","entity_name":"cuba-memorys","content":"cuba-memorys is written in Rust and stores everything in PostgreSQL 18 with pgvector.","observation_type":"fact","source":"agent"}' \
  "And a plain fact — remember this one for the verification below"
sleep 0.3

# ── 2. Retrieval ─────────────────────────────────────────────────────────────
echo -e "\n${BOLD}── Hybrid search: text + BM25 + vector, fused ──${NC}"
call faro '{"query":"what database does it use","limit":2}' \
  "compact is the default: 40% fewer tokens, same nDCG"
sleep 0.3

echo -e "\n${BOLD}── Anti-hallucination: a judge reads the evidence ──${NC}"
call faro '{"query":"cuba-memorys is written in Java","mode":"verify"}' \
  "A FALSE claim → contradicted (the store says Rust)"
call faro '{"query":"the best paella recipe uses saffron","mode":"verify"}' \
  "An UNRELATED claim → unknown, and no invented evidence"
sleep 0.3

# ── 3. Procedural memory: the fourth kind ────────────────────────────────────
echo -e "\n${BOLD}── Procedural memory: how things are DONE here ──${NC}"
call receta '{"action":"add","name":"publish a release","steps":["run the pre-flight: fmt, clippy, tests, audit","verify every file that holds a version agrees","tag — the tag is what publishes"],"context":"cuba-memorys"}' \
  "A recipe (ACT-R: declarative memory is reinforced by ACCESS, procedural by SUCCESS)"
call receta '{"action":"outcome","name":"publish a release","success":true}' \
  "It worked. Ranked by Wilson lower bound, so 1/1 does not outrank 47/50."
sleep 0.3

# ── 4. The CLI: your memory, without an LLM in the middle ────────────────────
echo -e "\n${BOLD}── And you can read it yourself ──${NC}"
cli "search from the shell — no agent, no tokens" search "embedding model" --limit 2
cli "what is in there, at a glance" dashboard
sleep 0.3

# ── 5. Operability ───────────────────────────────────────────────────────────
echo -e "\n${BOLD}── It tells you when something is wrong ──${NC}"
cli "schema, dimensions, config coherence, stale processes" doctor
sleep 0.3

echo -e "\n${GREEN}${BOLD}"
echo "  ✓ 28 MCP tools · 234 tests · 0 clippy warnings · cargo audit clean"
echo "  github.com/LeandroPG19/cuba-memorys"
echo -e "${NC}"
sleep 1.5
