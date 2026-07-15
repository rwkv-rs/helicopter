from __future__ import annotations

from scoreboard_server.cores.leaderboard import build_meta_payload
from scoreboard_server.db.repository import ScoreboardStore
from scoreboard_server.dtos.api.meta import MetaResponse


async def meta_response(store: ScoreboardStore) -> MetaResponse:
    entries = await store.list_latest_scores_for_space()
    return build_meta_payload(entries)
