from __future__ import annotations

from fastapi import FastAPI

from . import cancel, draft, options, pause, resume, start, status


def register(app: FastAPI) -> None:
    options.register(app)
    draft.register(app)
    status.register(app)
    start.register(app)
    pause.register(app)
    resume.register(app)
    cancel.register(app)
