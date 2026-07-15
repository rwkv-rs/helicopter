from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .api_v1 import router
from .auth import Authenticator
from .db.connection import Database, database_url_from_env
from .db.migrations import migration_state
from .errors import install_error_handlers
from .run_application import RunApplication
from .schema_application import SchemaApplication
from .settings import ScoreboardSettings


def create_app(
    settings: ScoreboardSettings | None = None, *, database_url: str | None = None
) -> FastAPI:
    resolved = settings or ScoreboardSettings.from_env()
    database = Database(database_url or database_url_from_env())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await database.open()
        try:
            app.state.schema_state = await migration_state(database)
            yield
        finally:
            await database.close()

    app = FastAPI(title="Helicopter Scoreboard", version="1.0.0", lifespan=lifespan)
    app.state.authenticator = Authenticator(resolved.tokens)
    app.state.database = database
    app.state.runs = RunApplication(database)
    app.state.schema = SchemaApplication(database)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Idempotency-Key", "If-Match"],
        expose_headers=["ETag", "X-Request-ID"],
    )

    @app.middleware("http")
    async def request_identity(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID") or uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    install_error_handlers(app)
    app.include_router(router)
    return app


app = create_app()
