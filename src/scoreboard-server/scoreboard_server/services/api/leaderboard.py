from __future__ import annotations

from scoreboard_server.cores.leaderboard import build_leaderboard_payload
from scoreboard_server.db.repository import ScoreboardStore
from scoreboard_server.dtos.api.leaderboard import LeaderboardResponse


async def leaderboard_response(
    store: ScoreboardStore,
    *,
    model: str | None,
    view: str,
) -> LeaderboardResponse:
    entries = await store.list_latest_scores_for_space()
    return build_leaderboard_payload(entries, selected_model=model, view=view)
