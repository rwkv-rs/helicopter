from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, AsyncIterator, Sequence
from urllib.parse import urlsplit

import aiosqlite
import asyncpg


class DatabaseConfigurationError(ValueError):
    pass


def database_url_from_env() -> str:
    return os.environ.get("SCOREBOARD_DATABASE_URL", "sqlite:///./scoreboard.db")


@dataclass(frozen=True, slots=True)
class DatabaseTarget:
    backend: str
    value: str

    @classmethod
    def parse(cls, url: str) -> "DatabaseTarget":
        if url.startswith("sqlite:///"):
            path = url.removeprefix("sqlite:///")
            if not path:
                raise DatabaseConfigurationError(
                    "SQLite database path must not be empty"
                )
            return cls("sqlite", str(Path(path).resolve()))
        parsed = urlsplit(url)
        if (
            parsed.scheme in {"postgres", "postgresql"}
            and parsed.hostname
            and parsed.path.strip("/")
        ):
            return cls("postgres", url)
        raise DatabaseConfigurationError(
            "SCOREBOARD_DATABASE_URL must be sqlite:///PATH or PostgreSQL URL"
        )


class DatabaseSession:
    def __init__(self, backend: str, connection: Any) -> None:
        self.backend = backend
        self.connection = connection

    def _sql(self, sql: str) -> str:
        if self.backend == "sqlite":
            return sql
        return _postgres_placeholders(sql)

    async def execute(self, sql: str, parameters: Sequence[Any] = ()) -> None:
        if self.backend == "sqlite":
            await self.connection.execute(sql, parameters)
        else:
            await self.connection.execute(self._sql(sql), *parameters)

    async def fetchone(
        self, sql: str, parameters: Sequence[Any] = ()
    ) -> dict[str, Any] | None:
        if self.backend == "sqlite":
            cursor = await self.connection.execute(sql, parameters)
            row = await cursor.fetchone()
            return dict(row) if row is not None else None
        row = await self.connection.fetchrow(self._sql(sql), *parameters)
        return dict(row) if row is not None else None

    async def fetchall(
        self, sql: str, parameters: Sequence[Any] = ()
    ) -> list[dict[str, Any]]:
        if self.backend == "sqlite":
            cursor = await self.connection.execute(sql, parameters)
            return [dict(row) for row in await cursor.fetchall()]
        return [
            dict(row)
            for row in await self.connection.fetch(self._sql(sql), *parameters)
        ]


class Database:
    def __init__(self, url: str) -> None:
        self.target = DatabaseTarget.parse(url)
        self._pool: asyncpg.Pool | None = None

    async def open(self) -> None:
        if self.target.backend == "postgres" and self._pool is None:
            self._pool = await asyncpg.create_pool(
                self.target.value,
                min_size=1,
                max_size=10,
                server_settings={"search_path": "public"},
            )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @asynccontextmanager
    async def transaction(
        self, *, immediate: bool = False
    ) -> AsyncIterator[DatabaseSession]:
        if self.target.backend == "sqlite":
            Path(self.target.value).parent.mkdir(parents=True, exist_ok=True)
            connection = await aiosqlite.connect(
                self.target.value, isolation_level=None
            )
            connection.row_factory = aiosqlite.Row
            await connection.execute("PRAGMA foreign_keys = ON")
            await connection.execute("PRAGMA busy_timeout = 30000")
            await connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            try:
                yield DatabaseSession("sqlite", connection)
            except BaseException:
                await connection.rollback()
                raise
            else:
                await connection.commit()
            finally:
                await connection.close()
            return
        await self.open()
        assert self._pool is not None
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                yield DatabaseSession("postgres", connection)

    @asynccontextmanager
    async def read(self) -> AsyncIterator[DatabaseSession]:
        if self.target.backend == "sqlite":
            connection = await aiosqlite.connect(self.target.value)
            connection.row_factory = aiosqlite.Row
            await connection.execute("PRAGMA foreign_keys = ON")
            try:
                yield DatabaseSession("sqlite", connection)
            finally:
                await connection.close()
            return
        await self.open()
        assert self._pool is not None
        async with self._pool.acquire() as connection:
            yield DatabaseSession("postgres", connection)


def _postgres_placeholders(sql: str) -> str:
    """Translate contract placeholders without touching SQL strings or comments."""

    output: list[str] = []
    index = 1
    position = 0
    quote: str | None = None
    line_comment = False
    block_comment = False
    while position < len(sql):
        char = sql[position]
        following = sql[position + 1] if position + 1 < len(sql) else ""
        if line_comment:
            output.append(char)
            if char == "\n":
                line_comment = False
        elif block_comment:
            output.append(char)
            if char == "*" and following == "/":
                output.append(following)
                position += 1
                block_comment = False
        elif quote is not None:
            output.append(char)
            if char == quote:
                if following == quote:
                    output.append(following)
                    position += 1
                else:
                    quote = None
        elif char in {"'", '"'}:
            quote = char
            output.append(char)
        elif char == "-" and following == "-":
            output.extend((char, following))
            position += 1
            line_comment = True
        elif char == "/" and following == "*":
            output.extend((char, following))
            position += 1
            block_comment = True
        elif char == "?":
            output.append(f"${index}")
            index += 1
        else:
            output.append(char)
        position += 1
    if quote is not None or block_comment:
        raise ValueError("unterminated SQL literal or comment")
    return "".join(output)
