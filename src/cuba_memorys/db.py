import asyncio
import os
import re
import sys
from datetime import datetime, date
from importlib import resources
from typing import Any
from urllib.parse import urlparse, urlunparse
from uuid import UUID

import asyncpg
import orjson

_DB_NAME_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,62}$")

_pool: asyncpg.Pool | None = None
_pool_lock = asyncio.Lock()
_semaphore = asyncio.Semaphore(8)

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
        print(f"[cuba-memorys] Cannot connect to PostgreSQL: {e}", file=sys.stderr)
        return

    try:
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1",
            target_db,
        )
        if not exists:
            if not _DB_NAME_PATTERN.match(target_db):
                print(f"[cuba-memorys] Invalid database name: {target_db}", file=sys.stderr)
                return
            await conn.execute(f'CREATE DATABASE "{target_db}"')
            print(f"[cuba-memorys] Created database '{target_db}'", file=sys.stderr)
        else:
            print(f"[cuba-memorys] Database '{target_db}' OK", file=sys.stderr)
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
        )
        return _pool

async def init_schema() -> None:
    pool = await get_pool()
    schema_sql = resources.files("cuba_memorys").joinpath("schema.sql").read_text()
    async with pool.acquire() as conn:
        await conn.execute(schema_sql)
    print("[cuba-memorys] Schema initialized (v1.0.1)", file=sys.stderr)

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
        await _pool.close()
        _pool = None

def _json_default(obj: Any) -> Any:
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type is not JSON serializable: {type(obj)}")

def serialize(obj: Any) -> str:
    return orjson.dumps(
        obj,
        default=_json_default,
        option=orjson.OPT_NON_STR_KEYS | orjson.OPT_NAIVE_UTC,
    ).decode("utf-8")
