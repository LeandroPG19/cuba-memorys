"""MCP JSON-RPC protocol layer and event loop for cuba-memorys.

Handles stdin/stdout transport, request routing, and the
REM sleep background consolidation daemon.
"""
import asyncio
import json
import logging
import signal
import sys
import time
from typing import Any

from cuba_memorys import __version__, db, search
from cuba_memorys.constants import REM_IDLE_SECONDS, TOOL_DEFINITIONS
from cuba_memorys.handlers import HANDLERS

logger = logging.getLogger("cuba-memorys.protocol")


# ── Request Handler ─────────────────────────────────────────────────

async def handle_request(request: dict[str, Any]) -> dict[str, Any] | None:
    """Route a JSON-RPC request to the appropriate handler."""
    method = request.get("method", "")
    req_id = request.get("id")
    params = request.get("params", {})

    if method == "initialize":
        return _rpc_result(req_id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "cuba-memorys", "version": __version__},
        })

    if method == "notifications/initialized":
        await db.init_schema()
        tfidf_count = await db.rebuild_tfidf_index()
        logger.info("TF-IDF index built: %d documents", tfidf_count)
        return None

    if method == "tools/list":
        return _rpc_result(req_id, {"tools": TOOL_DEFINITIONS})

    if method == "tools/call":
        return await _handle_tool_call(req_id, params)

    if req_id is not None:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Unknown method: {method}"},
        }

    return None


def _rpc_result(req_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    """Build a JSON-RPC success response."""
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


async def _handle_tool_call(
    req_id: Any, params: dict[str, Any],
) -> dict[str, Any]:
    """Dispatch a tool call to its handler."""
    tool_name = params.get("name", "")
    tool_args = params.get("arguments", {})

    handler = HANDLERS.get(tool_name)
    if not handler:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{
                    "type": "text",
                    "text": db.serialize({"error": f"Unknown tool: {tool_name}"}),
                }],
                "isError": True,
            },
        }

    try:
        result_text = await handler(tool_args)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": result_text}],
                "isError": False,
            },
        }
    except Exception:
        logger.exception("Tool '%s' raised an exception", tool_name)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{
                    "type": "text",
                    "text": db.serialize({
                        "error": "internal_error",
                        "message": "Internal server error",
                    }),
                }],
                "isError": True,
            },
        }


# ── REM Sleep: Background Consolidation Daemon ─────────────────────

_rem_running: bool = False


async def _rem_consolidation() -> None:
    """Background memory consolidation inspired by REM sleep.

    Runs FSRS decay, prunes low-importance observations,
    rebuilds TF-IDF index, and optionally runs PageRank.
    """
    global _rem_running
    _rem_running = True

    try:
        logger.info("💤 REM sleep starting — consolidating memories...")

        # FSRS decay (shared with zafra.decay)
        decay_result = await db.execute(
            "UPDATE brain_observations SET "
            "importance = GREATEST(0.01, "
            "  importance * POWER("
            "    1.0 + EXTRACT(EPOCH FROM (NOW() - last_accessed)) / 86400.0 "
            "    / (9.0 * GREATEST(stability, 0.1)), -1"
            "  )) "
            "WHERE last_accessed < NOW() - INTERVAL '1 day'",
        )
        logger.info("  ✓ Decay applied: %s", decay_result)

        # Prune
        prune_result = await db.execute(
            "DELETE FROM brain_observations "
            "WHERE importance < 0.05 AND access_count < 2",
        )
        logger.info("  ✓ Pruned: %s", prune_result)

        # TF-IDF rebuild
        tfidf_count = await db.rebuild_tfidf_index()
        logger.info("  ✓ TF-IDF rebuilt: %d docs", tfidf_count)

        search.cache_clear()

        # PageRank (optional dependency)
        try:
            import networkx as nx  # type: ignore[import-untyped]
            from cuba_memorys.handlers import _build_brain_graph

            g, has_data = await _build_brain_graph(directed=True)
            if has_data:
                pr = nx.pagerank(g, alpha=0.85, weight="weight")
                for name, score in pr.items():
                    pr_norm = min(1.0, score * len(pr))
                    await db.execute(
                        "UPDATE brain_entities SET "
                        "importance = LEAST(1.0, 0.6 * $1 + 0.4 * importance), "
                        "updated_at = NOW() WHERE name = $2",
                        pr_norm, name,
                    )
                logger.info("  ✓ PageRank: %d entities updated", len(pr))
        except ImportError:
            logger.info("  ⚠ PageRank skipped (networkx not installed)")

        logger.info("💤 REM sleep complete — memory optimized")

    except Exception:
        logger.exception("REM consolidation error")
    finally:
        _rem_running = False


# ── Main Event Loop ─────────────────────────────────────────────────

async def main() -> None:
    """Run the MCP server over stdin/stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="[cuba-memorys] %(message)s",
        stream=sys.stderr,
    )

    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown_event.set)

    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin.buffer)

    write_transport, _write_protocol = await loop.connect_write_pipe(
        lambda: asyncio.Protocol(), sys.stdout.buffer,
    )

    last_activity = time.monotonic()
    rem_task: asyncio.Task[None] | None = None

    try:
        while not shutdown_event.is_set():
            try:
                line = await asyncio.wait_for(reader.readline(), timeout=1.0)
            except asyncio.TimeoutError:
                idle_seconds = time.monotonic() - last_activity
                if (
                    idle_seconds >= REM_IDLE_SECONDS
                    and not _rem_running
                    and (rem_task is None or rem_task.done())
                ):
                    rem_task = asyncio.create_task(_rem_consolidation())
                    last_activity = time.monotonic()
                continue
            if not line:
                break

            last_activity = time.monotonic()

            if rem_task is not None and not rem_task.done():
                rem_task.cancel()
                rem_task = None

            line_str = line.decode("utf-8", errors="replace").strip()
            if not line_str:
                continue

            try:
                request = json.loads(line_str)
            except json.JSONDecodeError:
                continue

            response = await handle_request(request)

            if response is not None:
                response_bytes = db.serialize(response).encode("utf-8") + b"\n"
                write_transport.write(response_bytes)

    except (ConnectionError, BrokenPipeError, EOFError):
        pass
    finally:
        if rem_task is not None and not rem_task.done():
            rem_task.cancel()
        logger.info("Shutting down gracefully...")
        await db.close()
        write_transport.close()
