from __future__ import annotations

import uuid

from fastapi import FastAPI, HTTPException, Query, status

from scoreboard_server.db.repository import ScoreboardRepository


def register(app: FastAPI, repository: ScoreboardRepository) -> None:
    @app.get("/api/evaluations")
    async def evaluations(limit: int = Query(default=1000, ge=1, le=5000)):
        return await repository.list_evaluations(limit=limit)

    @app.get("/api/evaluations/{evaluation_id}/samples")
    async def evaluation_samples(
        evaluation_id: uuid.UUID,
        limit: int = Query(default=10, ge=1, le=100),
    ):
        groups = await repository.sample_groups(evaluation_id, limit=limit)
        if groups is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "evaluation not found")
        return groups
