from __future__ import annotations

from fastapi import FastAPI

from scoreboard_server.db.repository import ScoreboardStore
from . import (
    admin,
    capture_page,
    eval_context,
    eval_records,
    evaluation_publications,
    health,
    leaderboard,
    meta,
    refresh,
    score_history,
)
from scoreboard_server.services.api.evaluation_publications import (
    EvaluationPublicationService,
)


def register_api_routes(
    app: FastAPI,
    store: ScoreboardStore,
    publication_service: EvaluationPublicationService,
) -> None:
    health.register(app)
    meta.register(app, store)
    refresh.register(app, store)
    capture_page.register(app)
    leaderboard.register(app, store)
    eval_records.register(app, store)
    eval_context.register(app, store)
    evaluation_publications.register(app, publication_service)
    score_history.register(app, store)
    admin.register(app)
