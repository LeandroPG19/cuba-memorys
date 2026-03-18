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

import asyncpg
import networkx as nx  # type: ignore[import-untyped]

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

    v1.4.0: Reordered to PageRank→Decay→Prune→RelationDecay→TF-IDF (B5 fix).
    Added relation decay (E4) and decay idempotency via last_accessed (B1).
    """
    global _rem_running
    _rem_running = True

    try:
        logger.info("💤 REM sleep starting — consolidating memories...")

        # PageRank FIRST (B5 fix: uses pre-decay importance values)
        await _rem_pagerank()

        # FSRS decay with idempotency fix (B1: adds last_accessed = NOW())
        # V6: Session-aware — protect observations linked to active session goals
        decay_result = await db.execute(
            "UPDATE brain_observations SET "
            "importance = GREATEST(0.01, "
            "  importance * POWER("
            "    1.0 + EXTRACT(EPOCH FROM (NOW() - last_accessed)) / 86400.0 "
            "    / (9.0 * GREATEST(stability, 0.1)), -1"
            "  )), "
            "last_accessed = NOW() "
            "WHERE last_accessed < NOW() - INTERVAL '1 day' "
            "  AND entity_id NOT IN ("
            "    SELECT e.id FROM brain_entities e "
            "    JOIN brain_sessions s ON s.ended_at IS NULL "
            "    WHERE e.name ILIKE ANY("
            "      SELECT '%' || unnest(ARRAY(SELECT jsonb_array_elements_text(s.goals))) || '%'"
            "    )"
            "  )",
        )
        logger.info("  ✓ Decay applied: %s", decay_result)

        # Prune
        prune_result = await db.execute(
            "DELETE FROM brain_observations "
            "WHERE importance < 0.05 AND access_count < 2",
        )
        logger.info("  ✓ Pruned: %s", prune_result)

        # E4: Relation decay — weaken unused relations (half-life: 60 days)
        # Formula: strength' = strength × e^(-λ × days), λ = ln(2)/60
        rel_decay = await db.execute(
            "UPDATE brain_relations SET "
            "strength = GREATEST(0.01, "
            "  strength * EXP(-0.01155 * "
            "    EXTRACT(EPOCH FROM (NOW() - last_traversed)) / 86400.0"
            "  )) "
            "WHERE last_traversed < NOW() - INTERVAL '7 days'",
        )
        logger.info("  ✓ Relation decay: %s", rel_decay)

        # TF-IDF rebuild
        tfidf_count = await db.rebuild_tfidf_index()
        logger.info("  ✓ TF-IDF rebuilt: %d docs", tfidf_count)

        search.cache_clear()
        # V10: Clear embedding LRU cache alongside search cache
        from cuba_memorys import embeddings as _emb
        _emb.clear_cache()
        logger.info("💤 REM sleep complete — memory optimized")

    except asyncpg.PostgresError:
        logger.exception("REM consolidation DB error")
    except Exception:
        logger.exception("REM consolidation error")
    finally:
        _rem_running = False


async def _rem_pagerank() -> None:
    """Run PageRank as part of REM consolidation."""
    try:
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


# ── Main Event Loop ─────────────────────────────────────────────────

async def _process_message(
    line: bytes, write_transport: asyncio.WriteTransport,
) -> None:
    """Process a single JSON-RPC request line."""
    line_str = line.decode("utf-8", errors="replace").strip()
    if not line_str:
        return
    try:
        request = json.loads(line_str)
    except json.JSONDecodeError:
        return
    response = await handle_request(request)
    if response is not None:
        response_bytes = db.serialize(response).encode("utf-8") + b"\n"
        write_transport.write(response_bytes)


def _check_rem_idle(
    idle_seconds: float,
    rem_task: asyncio.Task[None] | None,
) -> asyncio.Task[None] | None:
    """Start REM consolidation if idle threshold exceeded."""
    if (
        idle_seconds >= REM_IDLE_SECONDS
        and not _rem_running
        and (rem_task is None or rem_task.done())
    ):
        return asyncio.create_task(_rem_consolidation())
    return rem_task


async def main() -> None:
    """Run the MCP server over stdin/stdout.

    CC reduced from 17 to 6 via extract-and-dispatch.
    """
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
                rem_task = _check_rem_idle(
                    time.monotonic() - last_activity, rem_task,
                )
                continue
            if not line:
                break

            last_activity = time.monotonic()
            if rem_task is not None and not rem_task.done():
                rem_task.cancel()
                rem_task = None

            await _process_message(line, write_transport)

    except (ConnectionError, BrokenPipeError, EOFError):
        pass
    finally:
        if rem_task is not None and not rem_task.done():
            rem_task.cancel()
        logger.info("Shutting down gracefully...")
        await db.close()
        write_transport.close()
