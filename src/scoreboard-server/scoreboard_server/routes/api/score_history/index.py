from __future__ import annotations

from fastapi import FastAPI, Query
from typing import Literal

from scoreboard_server.db.repository import ScoreboardStore
from scoreboard_server.dtos.api.score_history.index import ScoreHistoryResponse
from scoreboard_server.services.api.score_history.index import score_history_response


def register(app: FastAPI, store: ScoreboardStore) -> None:
    @app.get("/api/score-history")
    async def score_history(
        model: str = Query(...),
        benchmark: str = Query(...),
        scope: Literal["official", "non_official"] = "official",
    ) -> ScoreHistoryResponse:
        return await score_history_response(
            store, model=model, benchmark=benchmark, scope=scope
        )
