from __future__ import annotations

from datetime import datetime, timezone
import uuid

from fastapi import FastAPI, HTTPException, Query, status

from scoreboard_server.db.repository import ScoreboardRepository
from scoreboard_server.dtos.api.evaluation_results import AnswerOutcome


def register(app: FastAPI, repository: ScoreboardRepository) -> None:
    @app.get("/api/evaluations")
    async def evaluations(
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=1000, ge=1, le=5000),
        completed_before: datetime | None = Query(default=None),
    ):
        snapshot = completed_before or datetime.now(timezone.utc)
        if snapshot.tzinfo is None:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "completed_before must include a timezone",
            )
        return await repository.list_evaluations(
            completed_before=snapshot,
            offset=offset,
            limit=limit,
        )

    @app.get("/api/evaluations/{evaluation_id}/samples")
    async def evaluation_samples(
        evaluation_id: uuid.UUID,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=500),
        outcome: AnswerOutcome | None = Query(default=None),
    ):
        page = await repository.sample_page(
            evaluation_id,
            offset=offset,
            limit=limit,
            outcome=outcome,
        )
        if page is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "evaluation not found")
        return page
