from __future__ import annotations

from fastapi import FastAPI

from scoreboard_server.db.connection import Database


def register(app: FastAPI, database: Database) -> None:
    @app.get("/api/health")
    async def health() -> dict[str, str]:
        await database.require_pool().fetchval("SELECT 1")
        return {"status": "ok"}
