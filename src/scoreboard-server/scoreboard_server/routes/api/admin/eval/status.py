from __future__ import annotations

from fastapi import FastAPI

from scoreboard_server.dtos.api.admin.eval.status import AdminEvalStatusResponse
from scoreboard_server.services.api.admin.eval.status import admin_eval_status_response


def register(app: FastAPI) -> None:
    @app.get("/api/admin/eval/status")
    async def admin_status() -> AdminEvalStatusResponse:
        return admin_eval_status_response()
