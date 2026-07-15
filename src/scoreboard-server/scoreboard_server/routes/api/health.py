from __future__ import annotations

from fastapi import FastAPI

from scoreboard_server.dtos.api.health import HealthResponse
from scoreboard_server.services.api.health import health_response


def register(app: FastAPI) -> None:
    @app.get("/api/health")
    async def health() -> HealthResponse:
        return health_response()
