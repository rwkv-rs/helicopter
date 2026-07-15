from __future__ import annotations

from fastapi import FastAPI

from scoreboard_server.dtos.api.admin.health import AdminHealthResponse
from scoreboard_server.services.api.admin.health import admin_health_response


def register(app: FastAPI) -> None:
    @app.get("/api/admin/health")
    async def admin_health() -> AdminHealthResponse:
        return admin_health_response()
