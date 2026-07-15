from __future__ import annotations

from scoreboard_server.dtos.api.health import HealthResponse


def health_response() -> HealthResponse:
    return {"status": "ok"}
