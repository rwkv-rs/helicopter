from __future__ import annotations

from fastapi import FastAPI

from scoreboard_server.dtos.api.admin.eval.cancel import AdminEvalCancelResponse
from scoreboard_server.services.api.admin.eval.status import admin_eval_status_response


def register(app: FastAPI) -> None:
    @app.post("/api/admin/eval/cancel")
    async def admin_cancel() -> AdminEvalCancelResponse:
        return admin_eval_status_response()
