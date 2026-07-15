from __future__ import annotations

from tortoise import Tortoise

from .settings import DatabaseSettings
from .schema import apply_schema_sql


async def init_db(settings: DatabaseSettings | None = None, *, generate_schemas: bool = False) -> None:
    if Tortoise._inited:
        return
    resolved = settings or DatabaseSettings.from_env()
    await Tortoise.init(config=resolved.tortoise_config())
    if generate_schemas:
        await Tortoise.generate_schemas(safe=True)
        await apply_schema_sql()


async def close_db() -> None:
    if not Tortoise._inited:
        return
    await Tortoise.close_connections()
    Tortoise.apps.clear()
    Tortoise._inited = False
