"""PostgreSQL pool lifecycle and schema initialization."""

from __future__ import annotations

import json

import asyncpg

from .schema import SCHEMA_SQL
from .settings import DatabaseSettings


async def _configure_connection(connection: asyncpg.Connection) -> None:
    for type_name in ("json", "jsonb"):
        await connection.set_type_codec(
            type_name,
            schema="pg_catalog",
            encoder=lambda value: json.dumps(
                value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ),
            decoder=json.loads,
            format="text",
        )


class Database:
    def __init__(self, settings: DatabaseSettings):
        self.settings = settings
        self.pool: asyncpg.Pool | None = None

    async def start(self) -> None:
        if self.pool is not None:
            return
        self.pool = await asyncpg.create_pool(
            host=self.settings.host,
            port=self.settings.port,
            user=self.settings.user,
            password=self.settings.password,
            database=self.settings.database,
            min_size=self.settings.min_size,
            max_size=self.settings.max_size,
            init=_configure_connection,
        )
        try:
            async with self.pool.acquire() as connection:
                await connection.execute(SCHEMA_SQL)
        except Exception:
            await self.pool.close()
            self.pool = None
            raise

    async def stop(self) -> None:
        if self.pool is None:
            return
        await self.pool.close()
        self.pool = None

    def require_pool(self) -> asyncpg.Pool:
        if self.pool is None:
            raise RuntimeError("database is not started")
        return self.pool
