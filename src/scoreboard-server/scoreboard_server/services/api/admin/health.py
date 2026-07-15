from __future__ import annotations

from scoreboard_server.dtos.api.admin.health import AdminHealthResponse


def admin_health_response() -> AdminHealthResponse:
    return {"status": "disabled", "active": False, "auth_required": False}
