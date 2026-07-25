from fastapi import FastAPI

from scoreboard_server.db.connection import Database
from scoreboard_server.db.repository import ScoreboardRepository
from scoreboard_server.services.api.evaluation_publications import (
    EvaluationPublicationService,
)

from . import evaluation_publications, evaluation_results, health


def register_api_routes(
    app: FastAPI,
    *,
    database: Database,
    repository: ScoreboardRepository,
    publication_service: EvaluationPublicationService,
) -> None:
    health.register(app, database)
    evaluation_publications.register(app, publication_service)
    evaluation_results.register(app, repository)
