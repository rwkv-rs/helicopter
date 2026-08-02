from __future__ import annotations

from dataclasses import dataclass
import os


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


@dataclass(frozen=True, slots=True)
class DatabaseSettings:
    host: str = "127.0.0.1"
    port: int = 5432
    user: str = "postgres"
    password: str | None = None
    database: str = "helicopter_scoreboard"
    min_size: int = 1
    max_size: int = 10

    @classmethod
    def from_env(cls) -> "DatabaseSettings":
        min_size = max(1, _int_env("SCOREBOARD_DB_MIN_SIZE", 1))
        max_size = max(1, _int_env("SCOREBOARD_DB_MAX_SIZE", 10))
        if min_size > max_size:
            raise ValueError(
                "SCOREBOARD_DB_MIN_SIZE cannot exceed SCOREBOARD_DB_MAX_SIZE"
            )
        return cls(
            host=os.environ.get("SCOREBOARD_DB_HOST")
            or os.environ.get("PGHOST")
            or "127.0.0.1",
            port=_int_env("SCOREBOARD_DB_PORT", _int_env("PGPORT", 5432)),
            user=os.environ.get("SCOREBOARD_DB_USER")
            or os.environ.get("PGUSER")
            or "postgres",
            password=os.environ.get("SCOREBOARD_DB_PASSWORD")
            or os.environ.get("PGPASSWORD"),
            database=os.environ.get("SCOREBOARD_DB_NAME")
            or os.environ.get("PGDATABASE")
            or "helicopter_scoreboard",
            min_size=min_size,
            max_size=max_size,
        )
