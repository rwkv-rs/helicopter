from __future__ import annotations

from scoreboard_server.db.repository import ScoreboardStore
from scoreboard_server.dtos.api.eval_records import EvalRecordsResponse


async def eval_records_response(
    store: ScoreboardStore,
    *,
    task_id: int,
    only_wrong: bool,
    limit: int,
    offset: int,
) -> EvalRecordsResponse:
    records = await store.list_eval_records_for_space(
        task_id=str(task_id),
        only_wrong=only_wrong,
        limit=limit + 1,
        offset=offset,
        include_context=False,
        include_preview=True,
    )
    page = records[:limit]
    return {
        "task_id": task_id,
        "records": page,
        "offset": offset,
        "limit": limit,
        "next_offset": offset + len(page),
        "has_more": len(records) > limit,
    }
