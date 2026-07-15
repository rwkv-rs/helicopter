from __future__ import annotations

from fastapi import FastAPI, Query

from scoreboard_server.cores.normalize import EVAL_PAGE_SIZE
from scoreboard_server.db.repository import ScoreboardStore
from scoreboard_server.dtos.api.eval_records import EvalRecordsResponse
from scoreboard_server.services.api.eval_records import eval_records_response


def register(app: FastAPI, store: ScoreboardStore) -> None:
    @app.get("/api/eval-records")
    async def eval_records(
        task_id: int = Query(...),
        only_wrong: bool = Query(default=False),
        limit: int = Query(default=EVAL_PAGE_SIZE, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> EvalRecordsResponse:
        return await eval_records_response(
            store,
            task_id=task_id,
            only_wrong=only_wrong,
            limit=limit,
            offset=offset,
        )
