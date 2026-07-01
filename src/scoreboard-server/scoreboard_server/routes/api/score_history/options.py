from __future__ import annotations

from fastapi import FastAPI

from scoreboard_server.db.repository import ScoreboardStore
from scoreboard_server.dtos.api.score_history.options import ScoreHistoryOptionsResponse
from scoreboard_server.services.api.score_history.options import score_history_options_response


def register(app: FastAPI, store: ScoreboardStore) -> None:
    @app.get("/api/score-history/options")
    async def score_history_options() -> ScoreHistoryOptionsResponse:
        return await score_history_options_response(store)
