from __future__ import annotations

from typing import Any

from fastapi import Body, FastAPI, HTTPException

from scoreboard_server.dtos.api.capture_page import CapturePageResponse
from scoreboard_server.services.api.capture_page import capture_page_response


def register(app: FastAPI) -> None:
    @app.post("/api/capture-page")
    async def capture_page(payload: dict[str, Any] | None = Body(default=None)) -> CapturePageResponse:
        try:
            return await capture_page_response(payload)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"failed to save dashboard screenshot: {exc}") from exc
