"""Cuba-memorys MCP server — thin re-export module.

All logic has been decomposed into:
  - constants.py: tool definitions, thresholds, enums
  - handlers.py:  12 tool handler functions (CC-reduced)
  - protocol.py:  JSON-RPC transport, event loop, REM daemon
  - db.py:        database abstraction layer
  - search.py:    search queries + caching
  - embeddings.py: pgvector embedding helpers
  - hebbian.py:   Oja's rule + synapse weight boost
  - tfidf.py:     TF-IDF in-memory index

This file re-exports `main()` for backward compatibility with
the pyproject.toml entry point `cuba-memorys = "cuba_memorys.server:run"`.
"""
import asyncio

from cuba_memorys.protocol import main


def run() -> None:
    """Entry point for the MCP server (called by pyproject.toml)."""
    asyncio.run(main())
