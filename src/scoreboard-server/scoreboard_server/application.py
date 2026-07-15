from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from scoreboard_server.db.connection import close_db, init_db
from scoreboard_server.db.repository import ScoreboardStore
from scoreboard_server.db.settings import DatabaseSettings
from scoreboard_server.routes.api import register_api_routes


def create_app(settings: DatabaseSettings | None = None) -> FastAPI:
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
    register_api_routes(app, store)
    return app


app = create_app()
