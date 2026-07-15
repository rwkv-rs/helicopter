from __future__ import annotations

from .db.connection import Database
from .db.migrations import apply_migrations, migration_state
from .errors import DomainError


class SchemaApplication:
    """Own schema readiness and the authenticated migration use case."""

    def __init__(self, database: Database) -> None:
        self.database = database

    async def state(self) -> str:
        try:
            return await migration_state(self.database)
        except Exception as error:
            raise DomainError(
                "schema_unreachable", "database schema state could not be read", 503
            ) from error

    async def require_ready(self) -> None:
        state = await self.state()
        if state != "ready":
            raise DomainError(
                "schema_not_ready", f"database schema state is {state}", 503
            )

    async def migrate(self, *, subject: str) -> str:
        try:
            disposition = await apply_migrations(self.database, subject=subject)
        except ValueError as error:
            raise DomainError("schema_drift", str(error), 409) from error
        if await self.state() != "ready":
            raise DomainError(
                "schema_postcondition_failed",
                "database schema did not become ready",
                409,
            )
        return disposition
