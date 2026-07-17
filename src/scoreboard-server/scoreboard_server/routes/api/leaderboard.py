from __future__ import annotations

from typing import Literal

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
        scope: Literal["official", "non_official"] = "official",
    ) -> LeaderboardResponse:
        return await leaderboard_response(
            store,
            model=model,
            view=view,
            scope=scope,
        )
