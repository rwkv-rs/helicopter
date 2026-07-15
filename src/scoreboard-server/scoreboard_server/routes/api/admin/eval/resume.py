from __future__ import annotations

from fastapi import FastAPI

from scoreboard_server.dtos.api.admin.eval.resume import AdminEvalResumeResponse
from scoreboard_server.services.api.admin.eval.status import admin_eval_status_response


def register(app: FastAPI) -> None:
    @app.post("/api/admin/eval/resume")
    async def admin_resume() -> AdminEvalResumeResponse:
        return admin_eval_status_response()
