from __future__ import annotations

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .db.connection import Database
from .db.repository import ScoreboardRepository
from .db.settings import DatabaseSettings
from .routes.api import register_api_routes
from .services.api.evaluation_publications import (
    EvaluationPublicationService,
    publication_tokens_from_env,
)


def create_app(
    settings: DatabaseSettings | None = None,
    *,
    publication_tokens: dict[str, str] | None = None,
) -> FastAPI:
    resolved = settings or DatabaseSettings.from_env()
    database = Database(resolved)
    repository = ScoreboardRepository(database)
    tokens = (
        publication_tokens
        if publication_tokens is not None
        else publication_tokens_from_env(
            os.environ.get("SCOREBOARD_PUBLICATION_TOKENS", "{}")
        )
    )
    publication_service = EvaluationPublicationService(repository, tokens)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await database.start()
        try:
            yield
        finally:
            await database.stop()

    app = FastAPI(title="Helicopter Scoreboard", version="0.1.0", lifespan=lifespan)
    origins = [
        origin.strip()
        for origin in os.environ.get("SCOREBOARD_CORS_ORIGINS", "*").split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_api_routes(
        app,
        database=database,
        repository=repository,
        publication_service=publication_service,
    )
    app.state.database = database
    app.state.repository = repository
    return app


app = create_app()
