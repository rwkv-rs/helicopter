from __future__ import annotations

from fastapi import FastAPI

from scoreboard_server.db.repository import ScoreboardStore
from . import detail, index, options


def register(app: FastAPI, store: ScoreboardStore) -> None:
    options.register(app, store)
    index.register(app, store)
    detail.register(app, store)
