from __future__ import annotations

from typing import Any

from fastapi import Body, FastAPI, HTTPException

from scoreboard_server.services.api.admin.eval.status import SCHEDULER_CONTROL_ERROR


def register(app: FastAPI) -> None:
    @app.post("/api/admin/eval/start")
    async def admin_start(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
        _ = payload
        raise HTTPException(status_code=501, detail=SCHEDULER_CONTROL_ERROR)
