from __future__ import annotations

from scoreboard_server.db.repository import ScoreboardStore
from scoreboard_server.dtos.api.refresh import RefreshResponse


async def refresh_response(store: ScoreboardStore) -> RefreshResponse:
    entries = await store.list_latest_scores_for_space()
    return {"entry_count": len(entries), "errors": []}
