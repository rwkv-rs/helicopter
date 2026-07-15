from __future__ import annotations

from fastapi import FastAPI, Query

from scoreboard_server.db.repository import ScoreboardStore
from scoreboard_server.dtos.api.score_history.detail import ScoreHistoryDetailResponse
from scoreboard_server.services.api.score_history.detail import score_history_detail_response


def register(app: FastAPI, store: ScoreboardStore) -> None:
    @app.get("/api/score-history/detail")
    async def score_history_detail(task_id: int = Query(...)) -> ScoreHistoryDetailResponse:
        return await score_history_detail_response(store, task_id=task_id)
