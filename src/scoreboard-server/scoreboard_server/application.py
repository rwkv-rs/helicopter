from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from scoreboard_server.db.connection import close_db, init_db
from scoreboard_server.db.evaluation_publications import EvaluationPublicationRepository
from scoreboard_server.db.repository import ScoreboardStore
from scoreboard_server.db.settings import DatabaseSettings
from scoreboard_server.routes.api import register_api_routes
from scoreboard_server.services.api.evaluation_publications import (
    EvaluationPublicationService,
    TokenGrant,
    publication_grants_from_env,
)


def create_app(
    settings: DatabaseSettings | None = None,
    *,
    publication_grants: dict[str, TokenGrant] | None = None,
) -> FastAPI:
    resolved = settings or DatabaseSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await init_db(resolved, generate_schemas=True)
        try:
            yield
        finally:
            await close_db()

    app = FastAPI(title="Helicopter Scoreboard", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    store = ScoreboardStore(settings=resolved)
    publication_service = EvaluationPublicationService(
        EvaluationPublicationRepository(resolved),
        publication_grants
        if publication_grants is not None
        else publication_grants_from_env(),
    )
    register_api_routes(app, store, publication_service)
    return app


app = create_app()
