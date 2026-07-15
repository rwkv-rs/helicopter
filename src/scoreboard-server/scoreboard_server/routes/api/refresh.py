from __future__ import annotations

from fastapi import FastAPI

from scoreboard_server.db.repository import ScoreboardStore
from scoreboard_server.dtos.api.refresh import RefreshResponse
from scoreboard_server.services.api.refresh import refresh_response


def register(app: FastAPI, store: ScoreboardStore) -> None:
    @app.post("/api/refresh")
    async def refresh() -> RefreshResponse:
        return await refresh_response(store)
