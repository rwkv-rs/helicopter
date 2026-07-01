from __future__ import annotations

from fastapi import FastAPI, Query

from scoreboard_server.db.repository import ScoreboardStore
from scoreboard_server.dtos.api.eval_context import EvalContextResponse
from scoreboard_server.services.api.eval_context import eval_context_response


def register(app: FastAPI, store: ScoreboardStore) -> None:
    @app.get("/api/eval-context")
    async def eval_context(
        task_id: int = Query(...),
        sample_index: int = Query(...),
        repeat_index: int = Query(default=0),
        pass_index: int = Query(default=0),
    ) -> EvalContextResponse:
        return await eval_context_response(
            store,
            task_id=task_id,
            sample_index=sample_index,
            repeat_index=repeat_index,
            pass_index=pass_index,
        )
