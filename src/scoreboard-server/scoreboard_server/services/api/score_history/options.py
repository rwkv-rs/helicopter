from __future__ import annotations

from scoreboard_server.db.repository import ScoreboardStore
from scoreboard_server.dtos.api.score_history.options import ScoreHistoryOptionsResponse


async def score_history_options_response(
    store: ScoreboardStore, *, scope: str = "official"
) -> ScoreHistoryOptionsResponse:
    pairs = await store.list_score_history_pairs(is_tmp=scope == "non_official")
    return {
        "scope": scope,
        "models": sorted({pair["model"] for pair in pairs}),
        "benchmarks": sorted({pair["dataset"] for pair in pairs}),
        "pairs": pairs,
    }
