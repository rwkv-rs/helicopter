from __future__ import annotations

import asyncio

from scoreboard_server.adapters.screenshot import capture_page as capture_dashboard_page
from scoreboard_server.dtos.api.capture_page import CapturePageRequest, CapturePageResponse


async def capture_page_response(payload: CapturePageRequest | None) -> CapturePageResponse:
    resolved = payload or {}
    return await asyncio.to_thread(
        capture_dashboard_page,
        url=resolved.get("url"),
        width=resolved.get("width"),
        height=resolved.get("height"),
    )
