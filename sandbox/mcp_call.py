#!/usr/bin/env python3
"""Cliente MCP de un disparo: llama a una tool del binario del sandbox e imprime el resultado.

Arranca el servidor por stdio, hace el handshake, llama la tool y sale. El entorno
(DATABASE_URL, ONNX_MODEL_PATH, CUBA_*) se hereda del shell — hacé `source sandbox/env.sh` antes.

Uso:
    python3 sandbox/mcp_call.py cuba_faro '{"query":"decay","limit":3}'
    python3 sandbox/mcp_call.py --list          # lista las tools que expone el binario
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

BIN = os.environ.get("CUBA_SANDBOX_BIN")
PROD_PORT = ":5488/"


def guard() -> None:
    """Nunca dejar que este cliente hable con la base viva."""
    url = os.environ.get("DATABASE_URL", "")
    if PROD_PORT in url:
        sys.exit(f"ABORTADO: DATABASE_URL apunta a producción ({url})")
    if not BIN or not os.path.exists(BIN):
        sys.exit(
            f"ABORTADO: no existe el binario del sandbox: {BIN}\n  ¿corriste `source sandbox/env.sh` y `cargo build --release`?"
        )
    if "/cuba-memorys/rust/target" in BIN:
        sys.exit("ABORTADO: el binario es el del checkout de producción.")


def send(p: subprocess.Popen, obj: dict) -> None:
    p.stdin.write(json.dumps(obj) + "\n")
    p.stdin.flush()


def read_until(p: subprocess.Popen, want_id: int) -> dict | None:
    while True:
        line = p.stdout.readline()
        if not line:
            return None
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("id") == want_id:
            return msg


def main() -> int:
    guard()
    list_only = sys.argv[1] == "--list"
    tool = None if list_only else sys.argv[1]
    args = json.loads(sys.argv[2]) if (not list_only and len(sys.argv) > 2) else {}

    proc = subprocess.Popen(
        [BIN],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=open("/tmp/cuba-sandbox-mcp.stderr.log", "w"),
        text=True,
        env=dict(os.environ),
        bufsize=1,
    )
    send(
        proc,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "cuba-sandbox", "version": "0.1"},
            },
        },
    )
    if read_until(proc, 1) is None:
        print(
            "ERROR: el servidor murió durante initialize; ver /tmp/cuba-sandbox-mcp.stderr.log",
            file=sys.stderr,
        )
        return 1
    send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})

    if list_only:
        send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        msg = read_until(proc, 2) or {}
        tools = (msg.get("result") or {}).get("tools", [])
        print(f"{len(tools)} tools:")
        for t in tools:
            print(f"  {t['name']:<24} {t.get('description', '')[:80]}")
    else:
        send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": tool, "arguments": args},
            },
        )
        msg = read_until(proc, 2)
        if msg is None:
            print(
                "ERROR: el servidor murió antes de responder; ver /tmp/cuba-sandbox-mcp.stderr.log",
                file=sys.stderr,
            )
            return 1
        if "error" in msg:
            print(json.dumps(msg["error"], ensure_ascii=False, indent=2))
            return 1
        content = (msg.get("result") or {}).get("content", [])
        print(
            content[0]["text"]
            if content and "text" in content[0]
            else json.dumps(msg.get("result"), ensure_ascii=False)
        )

    proc.stdin.close()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
