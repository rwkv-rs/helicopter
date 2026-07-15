from __future__ import annotations

from fastapi import FastAPI

from scoreboard_server.dtos.api.admin.eval.draft import AdminEvalDraftResponse
from scoreboard_server.services.api.admin.eval.draft import admin_eval_draft_response


def register(app: FastAPI) -> None:
    @app.get("/api/admin/eval/draft")
    async def admin_draft() -> AdminEvalDraftResponse:
        return admin_eval_draft_response()
