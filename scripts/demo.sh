#!/bin/bash
# Demo script for cuba-memorys v0.7.0
# Records a terminal session showing core MCP tools
set -e

BINARY="./rust/target/release/cuba-memorys"
BLUE='\033[1;34m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
CYAN='\033[1;36m'
DIM='\033[2m'
BOLD='\033[1m'
NC='\033[0m'

# Helper: send JSON-RPC and pretty-print result
call() {
    local tool="$1"
    local args="$2"
    local label="$3"

    echo -e "\n${CYAN}>>> ${label}${NC}"
    echo -e "${DIM}cuba_${tool}(${args})${NC}"

    RESULT=$(printf '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"demo","version":"1.0"}}}\n{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"cuba_%s","arguments":%s}}\n' "$tool" "$args" | timeout 10 "$BINARY" 2>/dev/null | tail -1)

    echo "$RESULT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    r = d.get('result', d.get('error', {}))
    if 'content' in r:
        t = json.loads(r['content'][0]['text'])
        print(json.dumps(t, indent=2, ensure_ascii=False)[:600])
    else:
        print(json.dumps(r, indent=2)[:400])
except: print('(no output)')
" 2>/dev/null
    sleep 0.3
}

clear
echo -e "${BOLD}${GREEN}"
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║  cuba-memorys v0.7.0 — Persistent Memory for AI    ║"
echo "  ║  19 tools • Knowledge Graph • Sub-millisecond       ║"
echo "  ╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"
sleep 1.5

echo -e "${YELLOW}# Install:${NC} pip install cuba-memorys"
echo -e "${YELLOW}# Or:${NC}      npm install cuba-memorys"
sleep 1.5

echo -e "\n${BOLD}── Knowledge Graph ──${NC}"
call "alma" '{"action":"create","name":"FastAPI","entity_type":"technology"}' "Create entity"
sleep 0.5

call "cronica" '{"action":"add","entity_name":"FastAPI","content":"All endpoints must use async def with response_model for auto-documentation","observation_type":"lesson","source":"agent"}' "Store a lesson"
sleep 0.5

echo -e "\n${BOLD}── Hybrid Search (RRF + Entropy Routing) ──${NC}"
call "faro" '{"query":"async endpoints API documentation","format":"compact","limit":3}' "Search memory"
sleep 0.5

echo -e "\n${BOLD}── Anti-Hallucination Verification ──${NC}"
call "faro" '{"query":"FastAPI uses Django ORM","mode":"verify"}' "Verify claim"
sleep 0.5

echo -e "\n${BOLD}── Graph Analytics ──${NC}"
call "vigia" '{"metric":"summary"}' "Knowledge graph stats"
sleep 0.5

call "vigia" '{"metric":"bridges"}' "Bridge entities (betweenness centrality)"
sleep 0.5

echo -e "\n${BOLD}── Memory Maintenance ──${NC}"
call "zafra" '{"action":"decay"}' "Stratified exponential decay"
sleep 0.5

echo -e "\n${GREEN}${BOLD}"
echo "  Done! 19 tools available. Add to Claude Code:"
echo "    claude mcp add cuba-memorys -- cuba-memorys"
echo -e "${NC}"
sleep 2
