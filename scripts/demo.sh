#!/bin/bash
# Terminal demo for README assets/demo.gif — cuba-memorys v0.10.0
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINARY="${CUBA_BINARY_PATH:-$ROOT/rust/target/release/cuba-memorys}"
export DATABASE_URL="${DATABASE_URL:-postgresql://cuba:memorys2026@127.0.0.1:5488/brain}"
export CUBA_JUDGE="${CUBA_JUDGE:-heuristic}"

BLUE='\033[1;34m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
CYAN='\033[1;36m'
MAGENTA='\033[1;35m'
DIM='\033[2m'
BOLD='\033[1m'
NC='\033[0m'

if [[ ! -x "$BINARY" ]]; then
  echo "Binary not found: $BINARY (run: cd rust && cargo build --release)"
  exit 1
fi

call() {
  local tool="$1"
  local args="$2"
  local label="$3"

  echo -e "\n${CYAN}>>> ${label}${NC}"
  echo -e "${DIM}cuba_${tool}(${args})${NC}"

  RESULT=$(
    printf '%s\n' \
      '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"demo","version":"0.10"}}}' \
      "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"cuba_${tool}\",\"arguments\":${args}}}" \
      | timeout 15 "$BINARY" 2>/dev/null | tail -1
  )

  echo "$RESULT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    r = d.get('result', d.get('error', {}))
    if 'content' in r:
        t = json.loads(r['content'][0]['text'])
        print(json.dumps(t, indent=2, ensure_ascii=False)[:520])
    else:
        print(json.dumps(r, indent=2, ensure_ascii=False)[:400])
except Exception:
    print('(no output)')
" 2>/dev/null || echo "(timeout)"
  sleep 0.35
}

clear
echo -e "${BOLD}${GREEN}"
cat <<'BANNER'
  ╔══════════════════════════════════════════════════════════╗
  ║  cuba-memorys v0.10.0 — Persistent Memory for AI Agents  ║
  ║  25 tools • Bitemporal facts • Graph metrics • BM25 RRF  ║
  ╚══════════════════════════════════════════════════════════╝
BANNER
echo -e "${NC}"
sleep 1.2

echo -e "${YELLOW}# PyPI:${NC}  pip install cuba-memorys==1.12.0"
echo -e "${YELLOW}# npm:${NC}   npm install -g cuba-memorys@0.10.0"
echo -e "${MAGENTA}# MCP:${NC}   claude mcp add cuba-memorys -- cuba-memorys"
sleep 1.2

echo -e "\n${BOLD}── Knowledge graph + bitemporal mirror ──${NC}"
call alma '{"action":"create","name":"cuba-memorys","entity_type":"project"}' \
  "Create entity"
call cronica '{"action":"add","entity_name":"cuba-memorys","content":"v0.10 mirrors observations into brain_facts (bitemporal default on)","observation_type":"fact","source":"agent"}' \
  "Store fact (→ brain_facts)"
sleep 0.3

echo -e "\n${BOLD}── Hybrid search (RRF + BM25 + vector) ──${NC}"
call faro '{"query":"postgres MCP memory knowledge graph","format":"compact","limit":3}' \
  "Search (compact)"
call faro '{"query":"FastAPI uses Django ORM","mode":"verify"}' \
  "Verify claim (anti-hallucination)"
sleep 0.3

echo -e "\n${BOLD}── Graph intelligence (v0.10) ──${NC}"
call vigia '{"metric":"summary"}' "Graph summary"
call zafra '{"action":"pagerank"}' "PageRank → brain_node_metrics + energy"
call zafra '{"action":"communities"}' "Leiden communities (persist)"
sleep 0.3

echo -e "\n${BOLD}── Session lifecycle ──${NC}"
call jornada '{"action":"start","goals":["ship v0.10 README demo"]}' "Start session"
call jornada '{"action":"current"}' "Session context"
sleep 0.3

echo -e "\n${GREEN}${BOLD}"
echo "  ✓ 25 MCP tools • merge-gate: E2E 73 + live session 25"
echo "  Docs: github.com/LeandroPG19/cuba-memorys"
echo -e "${NC}"
sleep 1.5