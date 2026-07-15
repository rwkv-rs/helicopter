from __future__ import annotations

from fastapi import FastAPI, Query

from scoreboard_server.cores.normalize import DEFAULT_TABLE_VIEW
from scoreboard_server.db.repository import ScoreboardStore
from scoreboard_server.dtos.api.leaderboard import LeaderboardResponse
from scoreboard_server.services.api.leaderboard import leaderboard_response


def register(app: FastAPI, store: ScoreboardStore) -> None:
    @app.get("/api/leaderboard")
    async def leaderboard(
        model: str | None = Query(default=None),
        view: str = Query(default=DEFAULT_TABLE_VIEW),
    ) -> LeaderboardResponse:
        return await leaderboard_response(store, model=model, view=view)
