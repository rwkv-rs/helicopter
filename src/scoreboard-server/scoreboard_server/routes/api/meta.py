from __future__ import annotations

from fastapi import FastAPI

from scoreboard_server.db.repository import ScoreboardStore
from scoreboard_server.dtos.api.meta import MetaResponse
from scoreboard_server.services.api.meta import meta_response


def register(app: FastAPI, store: ScoreboardStore) -> None:
    @app.get("/api/meta")
    async def meta() -> MetaResponse:
        return await meta_response(store)
