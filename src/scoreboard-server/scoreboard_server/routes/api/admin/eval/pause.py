from __future__ import annotations

from fastapi import FastAPI

from scoreboard_server.dtos.api.admin.eval.pause import AdminEvalPauseResponse
from scoreboard_server.services.api.admin.eval.status import admin_eval_status_response


def register(app: FastAPI) -> None:
    @app.post("/api/admin/eval/pause")
    async def admin_pause() -> AdminEvalPauseResponse:
        return admin_eval_status_response()
