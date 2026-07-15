from __future__ import annotations

from fastapi import FastAPI

from . import backpressure, eval, health


def register(app: FastAPI) -> None:
    health.register(app)
    eval.register(app)
    backpressure.register(app)
