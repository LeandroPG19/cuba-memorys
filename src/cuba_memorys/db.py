import asyncio
import logging
import os
import re
from datetime import datetime, date
from decimal import Decimal
from importlib import resources
from typing import Any
from urllib.parse import urlparse, urlunparse
from uuid import UUID

import asyncpg
import orjson

from cuba_memorys import __version__

logger = logging.getLogger("cuba-memorys.db")

_DB_NAME_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,62}$")

_pool: asyncpg.Pool | None = None
_pool_lock = asyncio.Lock()
_semaphore = asyncio.Semaphore(8)
_pgvector_available: bool = False

def has_pgvector() -> bool:
    """Check if pgvector extension is active in the database."""
    return _pgvector_available

def _get_admin_url_and_db() -> tuple[str, str]:
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError(
            "DATABASE_URL environment variable is required. "
            "Example: postgresql://user:pass@localhost:5432/brain"
        )

    parsed = urlparse(url)
    target_db = parsed.path.lstrip("/") or "brain"

    admin_parsed = parsed._replace(path="/postgres")
    admin_url = urlunparse(admin_parsed)

    return admin_url, target_db

async def _ensure_database_exists() -> None:
    admin_url, target_db = _get_admin_url_and_db()

    try:
        conn = await asyncpg.connect(admin_url, timeout=10)
    except (OSError, asyncpg.PostgresError) as e:
        logger.error("Cannot connect to PostgreSQL: %s", e)
        return

    try:
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1",
            target_db,
        )
        if not exists:
            if not _DB_NAME_PATTERN.match(target_db):
                logger.error("Invalid database name: %s", target_db)
                return
            await conn.execute(f'CREATE DATABASE "{target_db}"')
            logger.info("Created database '%s'", target_db)
        else:
            logger.info("Database '%s' OK", target_db)
    except asyncpg.DuplicateDatabaseError:
        pass
    finally:
        await conn.close()

async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is not None:
        return _pool

    async with _pool_lock:
        if _pool is not None:
            return _pool

        await _ensure_database_exists()

        database_url = os.environ.get("DATABASE_URL", "")
        if not database_url:
            raise RuntimeError(
                "DATABASE_URL environment variable is required. "
                "Example: postgresql://user:pass@localhost:5432/brain"
            )

        async def _init_connection(conn: asyncpg.Connection) -> None:
            await conn.execute("SELECT 1")

        _pool = await asyncpg.create_pool(
            database_url,
            min_size=2,
            max_size=10,
            command_timeout=30,
            statement_cache_size=512,
            init=_init_connection,
            setup=_init_connection,
        )
        return _pool

async def init_schema() -> None:
    global _pgvector_available
    pool = await get_pool()
    schema_sql = resources.files("cuba_memorys").joinpath("schema.sql").read_text()
    async with pool.acquire() as conn:
        await conn.execute(schema_sql)

        # Detect pgvector extension
        ext = await conn.fetchval(
            "SELECT 1 FROM pg_extension WHERE extname = 'vector'",
        )
        if ext:
            _pgvector_available = True
            # Migrate embedding column from float4[] to vector(384)
            col_type = await conn.fetchval(
                "SELECT data_type FROM information_schema.columns "
                "WHERE table_name = 'brain_observations' "
                "AND column_name = 'embedding'",
            )
            if col_type and col_type != "USER-DEFINED":
                await conn.execute(
                    "ALTER TABLE brain_observations "
                    "DROP COLUMN IF EXISTS embedding",
                )
                await conn.execute(
                    "ALTER TABLE brain_observations "
                    "ADD COLUMN embedding vector(384)",
                )
                logger.info("Migrated embedding column to vector(384)")
            elif not col_type:
                await conn.execute(
                    "ALTER TABLE brain_observations "
                    "ADD COLUMN IF NOT EXISTS embedding vector(384)",
                )
            # HNSW index for cosine similarity
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_obs_embedding_hnsw "
                "ON brain_observations USING hnsw (embedding vector_cosine_ops) "
                "WITH (m = 16, ef_construction = 64)",
            )
            logger.info("pgvector active — HNSW index ready (384d)")
        else:
            logger.info("pgvector not installed — using TF-IDF + trigrams")

    logger.info("Schema initialized (%s)", __version__)

async def execute(query: str, *args: Any) -> str:
    async with _semaphore:
        pool = await get_pool()
        async with pool.acquire() as conn:
            return await conn.execute(query, *args)

async def fetchrow(query: str, *args: Any) -> dict[str, Any] | None:
    async with _semaphore:
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None

async def fetch(query: str, *args: Any) -> list[dict[str, Any]]:
    async with _semaphore:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(r) for r in rows]

async def fetchval(query: str, *args: Any) -> Any:
    async with _semaphore:
        pool = await get_pool()
        async with pool.acquire() as conn:
            return await conn.fetchval(query, *args)

async def close() -> None:
    global _pool
    if _pool is not None:
        try:
            await asyncio.wait_for(_pool.close(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Pool close timed out, terminating")
            _pool.terminate()
        _pool = None


async def rebuild_tfidf_index() -> int:
    """Rebuild TF-IDF index from all observations.

    Returns:
        Number of documents indexed.
    """
    from cuba_memorys.tfidf import tfidf_index
    rows = await fetch(
        "SELECT content FROM brain_observations "
        "WHERE observation_type != 'superseded' "
        "AND (valid_until IS NULL OR valid_until > NOW())",
    )
    if rows:
        corpus = [r["content"] for r in rows]
        tfidf_index.fit(corpus)
    return len(rows)


async def rebuild_embeddings(batch_size: int = 50) -> int:
    """Backfill embeddings for observations that lack them.

    Requires pgvector + ONNX model.

    Returns:
        Number of observations updated.
    """
    from cuba_memorys import embeddings
    if not has_pgvector() or not embeddings.is_available():
        return 0

    rows = await fetch(
        "SELECT id, content FROM brain_observations "
        "WHERE embedding IS NULL "
        "AND observation_type != 'superseded' LIMIT 500",
    )
    if not rows:
        return 0

    updated = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        texts = [r["content"] for r in batch]
        vecs = embeddings.embed(texts)
        for j, vec in enumerate(vecs):
            await execute(
                "UPDATE brain_observations SET embedding = $1 WHERE id = $2",
                vec.tolist(), batch[j]["id"],
            )
            updated += 1
    return updated

def _json_default(obj: Any) -> Any:
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Type is not JSON serializable: {type(obj)}")

def serialize(obj: Any) -> str:
    return orjson.dumps(
        obj,
        default=_json_default,
        option=orjson.OPT_NON_STR_KEYS | orjson.OPT_NAIVE_UTC,
    ).decode("utf-8")
