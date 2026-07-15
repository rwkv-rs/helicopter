from __future__ import annotations

from fastapi import FastAPI

from scoreboard_server.dtos.api.admin.eval.options import AdminEvalOptionsResponse
from scoreboard_server.services.api.admin.eval.options import admin_eval_options_response


def register(app: FastAPI) -> None:
    @app.get("/api/admin/eval/options")
    async def admin_options() -> AdminEvalOptionsResponse:
        return admin_eval_options_response()
